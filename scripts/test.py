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

        # Handle single GPU vs multi-GPU
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample) 
            for sample in all_samples:
                all_images.append(sample.cpu().numpy())
        else:
            # Single GPU mode
            all_images.append(sample.cpu().numpy())
            
        logger.log(f"created {len(all_images) * args.batch_size} samples\n")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples] 
    arr = arr.reshape((arr.shape[0],arr.shape[2],arr.shape[3],arr.shape[4])) 

    # 根據輸入類型處理輸出
    if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
        logger.log("Processing TIFF output...")
        
        # 讀取原始 TIFF 嚟獲取原始尺寸
        try:
            original_data = tifffile.imread(args.base_samples)
            original_depth, original_height, original_width = original_data.shape
            logger.log(f"Original TIFF shape: {original_depth}x{original_height}x{original_width}")
        except Exception as e:
            logger.log(f"Warning: Could not read original TIFF for dimensions: {e}")
            original_depth, original_height, original_width = None, None, None
        
        if len(arr) == 1:
            # 單一 patch 處理
            logger.log("Processing single patch...")
            arr_result = arr[0]  # Shape: (H, W, D)
            
            # 如果有原始尺寸資訊，移除 padding
            if original_depth is not None and arr_result.shape[2] > original_depth:
                pad_before = (arr_result.shape[2] - original_depth) // 2
                arr_result = arr_result[:, :, pad_before:pad_before+original_depth]
                logger.log(f"Removed padding: final shape {arr_result.shape}")
                
        else:
            # 多個 patches，需要重組
            logger.log(f"Processing {len(arr)} patches for reconstruction...")
            
            if original_depth is not None:
                # 重組到原始尺寸
                arr_result = np.zeros((original_height, original_width, original_depth))
                
                # 根據 patch 策略重組（基於 load_data_for_worker 入面嘅邏輯）
                patch_depth = 2
                overlap = 1
                stride = patch_depth - overlap
                
                for i, patch in enumerate(arr):
                    start_idx = i * stride
                    end_idx = min(start_idx + patch_depth, original_depth)
                    actual_depth = end_idx - start_idx
                    
                    logger.log(f"Reconstructing patch {i}: slices {start_idx}-{end_idx}")
                    
                    if i == 0:
                        # 第一個 patch，直接複製
                        arr_result[:, :, start_idx:end_idx] = patch[:, :, :actual_depth]
                    elif i == len(arr) - 1:
                        # 最後一個 patch，處理可能嘅 padding
                        if actual_depth < patch_depth:
                            # 有 padding，只取有效部分
                            arr_result[:, :, start_idx:end_idx] = patch[:, :, :actual_depth]
                        else:
                            # 處理重疊
                            overlap_size = min(overlap, start_idx)
                            if overlap_size > 0:
                                # 重疊區域用平均值
                                overlap_start = start_idx
                                overlap_end = start_idx + overlap_size
                                existing = arr_result[:, :, overlap_start:overlap_end]
                                new_data = patch[:, :, :overlap_size]
                                arr_result[:, :, overlap_start:overlap_end] = (existing + new_data) / 2
                                
                                # 非重疊部分
                                if overlap_end < end_idx:
                                    arr_result[:, :, overlap_end:end_idx] = patch[:, :, overlap_size:actual_depth]
                            else:
                                arr_result[:, :, start_idx:end_idx] = patch[:, :, :actual_depth]
                    else:
                        # 中間 patches，處理兩邊重疊
                        overlap_size = min(overlap, start_idx)
                        if overlap_size > 0:
                            # 處理與前一個 patch 嘅重疊
                            overlap_start = start_idx
                            overlap_end = start_idx + overlap_size
                            existing = arr_result[:, :, overlap_start:overlap_end]
                            new_data = patch[:, :, :overlap_size]
                            arr_result[:, :, overlap_start:overlap_end] = (existing + new_data) / 2
                            
                            # 非重疊部分
                            arr_result[:, :, overlap_end:end_idx] = patch[:, :, overlap_size:actual_depth]
                        else:
                            arr_result[:, :, start_idx:end_idx] = patch[:, :, :actual_depth]
                
                logger.log(f"Reconstruction complete: final shape {arr_result.shape}")
            else:
                # 冇原始尺寸資訊，用第一個 patch
                arr_result = arr[0]
                logger.log(f"Warning: No original dimensions available, using first patch only")
                
    else:
        # 原來 NPZ 輸出處理
        logger.log("Processing NPZ output...")
        arr_result = np.zeros((192,288,576))     
        index = 0
        for i in range(6):
            arr_result[:, :, i*96:(i+1)*96] = arr[index,:,:,:]
            index += 1

    # Handle single GPU vs multi-GPU for saving
    if not dist.is_initialized() or dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{datetime.datetime.now().strftime('%H%M%S%f')}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr_result)
        
        # 如果係 TIFF 輸入，都保存一個 TIFF 輸出
        if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
            tiff_out_path = out_path.replace('.npz', '.tif')
            # 轉換維度從 (H,W,D) 到 (D,H,W) 用於 TIFF 保存
            tiff_data = arr_result.transpose(2, 0, 1)
            tifffile.imwrite(tiff_out_path, tiff_data.astype(np.float32))
            logger.log(f"Also saved as TIFF: {tiff_out_path}")

    # Only barrier if distributed is initialized
    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    if base_samples.endswith('.tif') or base_samples.endswith('.tiff'):
        # 讀取 TIFF 文件
        import tifffile
        
        # 讀取 3D TIFF
        low_pet = tifffile.imread(base_samples)  # Shape: (depth, height, width)
        
        # 獲取實際尺寸
        depth, height, width = low_pet.shape  # 例如 (105, 200, 200)
        logger.log(f'TIFF shape: {depth}x{height}x{width}')
        
        # 動態決定 patch 策略
        patch_depth = 2  # 可以調整呢個值
        
        if depth <= patch_depth:
            # 如果深度小於等於 patch_depth，用一個 patch，需要 padding
            image_arr = np.zeros((1, height, width, patch_depth))
            pad_before = (patch_depth - depth) // 2
            pad_after = patch_depth - depth - pad_before
            padded_volume = np.pad(low_pet, ((pad_before, pad_after), (0, 0), (0, 0)), 
                                  mode='constant', constant_values=0)
            image_arr[0, :, :, :] = padded_volume.transpose(1, 2, 0)  # (D,H,W) -> (H,W,D)
            logger.log(f'Using 1 patch with padding: {patch_depth} slices')
        else:
            # 動態計算需要幾多個 patches
            overlap = 1 # 重疊大小，可以調整
            stride = patch_depth - overlap
            num_patches = (depth - overlap + stride - 1) // stride  # 向上取整
            
            logger.log(f'Using {num_patches} patches with overlap {overlap}')
            image_arr = np.zeros((num_patches, height, width, patch_depth))
            
            for patch_idx in range(num_patches):
                start_idx = patch_idx * stride
                end_idx = min(start_idx + patch_depth, depth)
                actual_depth = end_idx - start_idx
                
                if actual_depth == patch_depth:
                    # 完整 patch
                    image_arr[patch_idx, :, :, :] = low_pet[start_idx:end_idx, :, :].transpose(1, 2, 0)
                else:
                    # 最後一個 patch 可能唔夠長，需要 padding
                    patch_data = np.zeros((height, width, patch_depth))
                    patch_data[:, :, :actual_depth] = low_pet[start_idx:end_idx, :, :].transpose(1, 2, 0)
                    image_arr[patch_idx, :, :, :] = patch_data
                    
                logger.log(f'Patch {patch_idx}: slices {start_idx}-{end_idx}')
                
    else:
        # 原來 NPZ 處理邏輯
        with bf.BlobFile(base_samples, "rb") as f:
            obj = np.load(f)
            low_pet = obj["arr_0"][0] 
            image_arr = np.zeros((6,192,288,96))
            index = 0
            for i in (0, 86, 172, 258, 344, 424):
                image_arr[index,:,:,:] = low_pet[:, :, i:i+96]
                index += 1
    
    # 數據正規化（對所有情況都適用）
    image_arr[image_arr>4] = 4
    image_arr = image_arr/4

    # Handle single GPU vs multi-GPU
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


if __name__ == "__main__":
    main()
