import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tifffile  
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import datetime

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    logger.log("creating samples...")
    all_images = []

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()} 

        shape = (args.batch_size, 1, model_kwargs['low_res'].shape[2], model_kwargs['low_res'].shape[3], model_kwargs['low_res'].shape[4])
        
        if device == "cuda":
            th.cuda.manual_seed_all(10)
        else:
            th.manual_seed(10)
        noise = th.randn(*shape, device=device)

        sample = diffusion.p_sample_loop(
            model,
            shape, 
            noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = sample.permute(0, 1, 3, 4, 2)
        sample = sample.contiguous() 

        // Handle single GPU vs multi-GPU
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample) 
            for sample in all_samples:
                all_images.append(sample.cpu().numpy())
        else:
            // Single GPU mode
            all_images.append(sample.cpu().numpy())
            
        logger.log(f"created {len(all_images) * args.batch_size} samples\n")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples] 
    arr = arr.reshape((arr.shape[0],arr.shape[2],arr.shape[3],arr.shape[4])) 

    // 根據輸入類型處理輸出
    if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
        logger.log("Processing TIFF output...")
        
        // 讀取原始 TIFF 嚟獲取原始尺寸
        try:
            original_data = tifffile.imread(args.base_samples)
            original_depth, original_height, original_width = original_data.shape
            logger.log(f"Original TIFF shape: {original_depth}x{original_height}x{original_width}")
        except Exception as e:
            logger.log(f"Warning: Could not read original TIFF for dimensions: {e}")
            original_depth, original_height, original_width = None, None, None
        
        if len(arr) == 1:
            // 單一 patch 處理
            logger.log("Processing single patch...")
            arr_result = arr[0]  # Shape: (H, W, D)
            
            // 如果有原始尺寸資訊，移除 padding
            if original_depth is not None and arr_result.shape[2] > original_depth:
                pad_before = (arr_result.shape[2] - original_depth) // 2
                arr_result = arr_result[:, :, pad_before:pad_before+original_depth]
                logger.log(f"Removed padding: final shape {arr_result.shape}")
                
        else:
            // 多個 patches，需要重組
            logger.log(f"Processing {len(arr)} patches for reconstruction...")
            
            if original_depth is not None:
                // 重組到原始尺寸
                arr_result = np.zeros((original_height, original_width, original_depth))
                count_arr = np.zeros_like(arr_result)
                
                // 生成與 load 時相同的 starts
                resolution = args.large_size  // 從命令行獲取
                x_starts = _calculate_xy_starts(original_height, resolution)
                y_starts = _calculate_xy_starts(original_width, resolution)
                z_starts = _calculate_z_starts(original_depth, resolution)
                
                patch_idx = 0
                for x_start in x_starts:
                    for y_start in y_starts:
                        for z_start in z_starts:
                            patch = arr[patch_idx]
                            x_end = min(x_start + resolution, original_height)
                            y_end = min(y_start + resolution, original_width)
                            z_end = min(z_start + resolution, original_depth)
                            
                            arr_result[x_start:x_end, y_start:y_end, z_start:z_end] += patch[0:x_end-x_start, 0:y_end-y_start, 0:z_end-z_start]
                            count_arr[x_start:x_end, y_start:y_end, z_start:z_end] += 1
                            patch_idx += 1
                
                arr_result = np.divide(arr_result, count_arr, where=count_arr != 0)
                
                logger.log(f"Reconstruction complete: final shape {arr_result.shape}")
            else:
                // 冇原始尺寸資訊，用第一個 patch
                arr_result = arr[0]
                logger.log(f"Warning: No original dimensions available, using first patch only")
                
    else:
        // 原來 NPZ 輸出處理
        logger.log("Processing NPZ output...")
        arr_result = np.zeros((192,288,576))     
        index = 0
        for i in range(6):
            arr_result[:, :, i*96:(i+1)*96] = arr[index,:,:,:]
            index += 1

    // Handle single GPU vs multi-GPU for saving
    if not dist.is_initialized() or dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{datetime.datetime.now().strftime('%H%M%S%f')}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr_result)
        
        // 如果係 TIFF 輸入，都保存一個 TIFF 輸出
        if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
            tiff_out_path = out_path.replace('.npz', '.tif')
            // 轉換維度從 (H,W,D) 到 (D,H,W) 用於 TIFF 保存
            tiff_data = arr_result.transpose(2, 0, 1)
            tifffile.imwrite(tiff_out_path, tiff_data.astype(np.float32))
            logger.log(f"Also saved as TIFF: {tiff_out_path}")

    // Only barrier if distributed is initialized
    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    if base_samples.endswith('.tif') or base_samples.endswith('.tiff'):
        vol = tifffile.imread(base_samples)
        if vol.ndim == 3:  # (D,H,W)
            D, H, W = vol.shape
        elif vol.ndim == 4 and vol.shape[0] >= 2:  # (C,D,H,W)
            _, D, H, W = vol.shape
        else:
            logger.log("Unsupported TIFF format")
            return

        resolution = 80  // 從訓練設定
        if H < resolution or W < resolution or D < resolution:
            logger.log(f"Volume too small ({H}x{W}x{D}), skipping")
            return

        // 正規化
        vol[vol > 4] = 4
        vol = vol / 4.0

        // 生成 patch 起始點
        x_starts = _calculate_xy_starts(H, resolution)
        y_starts = _calculate_xy_starts(W, resolution)
        z_starts = _calculate_z_starts(D, resolution)

        image_arr = []
        for x_start in x_starts:
            for y_start in y_starts:
                for z_start in z_starts:
                    x_end = min(x_start + resolution, H)
                    y_end = min(y_start + resolution, W)
                    z_end = min(z_start + resolution, D)
                    
                    patch = vol[z_start:z_end, x_start:x_end, y_start:y_end]
                    
                    // Padding 如果唔夠
                    padded_patch = np.zeros((resolution, resolution, resolution))
                    padded_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    image_arr.append(padded_patch.transpose(1, 2, 0))  // (H,W,D)

        image_arr = np.array(image_arr)
    else:
        // 原來 NPZ 處理邏輯
        with bf.BlobFile(base_samples, "rb") as f:
            obj = np.load(f)
            low_pet = obj["arr_0"][0] 
            image_arr = np.zeros((6,192,288,96))
            index = 0
            for i in (0, 86, 172, 258, 344, 424):
                image_arr[index,:,:,:] = low_pet[:, :, i:i+96]
                index += 1
    
    // 數據正規化（對所有情況都適用）
    image_arr[image_arr>4] = 4
    image_arr = image_arr/4

    // Handle single GPU vs multi-GPU
    if dist.is_initialized():
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
    else:
        rank = 0
        num_ranks = 1
        
    logger.log('rank:{%d}' % (rank))
    logger.log('num_ranks:{%d}' % (num_ranks))
    logger.log('total patches: %d' % len(image_arr))

    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            logger.log('rank:{%d}, i:{%d}, buffer_len:{%d}' % (rank, i, len(buffer)))

            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch.unsqueeze(0) 
                batch = batch.permute(0, 1, 4, 2, 3)
                logger.log('batch_shape:{%d,%d,%d,%d,%d}' % (batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]))
                res = dict(low_res=batch)
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def _calculate_xy_starts(dim_size, patch_size):
    overlap = 20
    stride = patch_size - overlap
    max_overlap = int(patch_size * 0.8)
    
    starts = [0]
    pos = stride
    while pos + patch_size <= dim_size:
        if starts:
            prev_end = starts[-1] + patch_size
            overlap_size = max(0, prev_end - pos)
            if overlap_size > max_overlap:
                pos += stride
                continue
        starts.append(pos)
        pos += stride
    
    if starts and starts[-1] + patch_size < dim_size:
        last_start = dim_size - patch_size
        if last_start > starts[-1]:
            prev_end = starts[-1] + patch_size
            overlap_size = max(0, prev_end - last_start)
            if overlap_size <= max_overlap:
                starts.append(last_start)
    
    return starts

def _calculate_z_starts(dim_size, patch_size):
    max_overlap = int(patch_size * 0.8)
    starts = [0]
    
    if dim_size > patch_size:
        second_start = dim_size - patch_size
        if second_start > 0:
            first_end = patch_size
            overlap_size = max(0, first_end - second_start)
            if overlap_size <= max_overlap:
                starts.append(second_start)
    
    return starts


if __name__ == "__main__":
    main()