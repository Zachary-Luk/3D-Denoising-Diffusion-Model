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
    data_generator = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond, args.large_size)
    logger.log("creating samples...")
    all_images = []

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    for model_kwargs in data_generator:
        if model_kwargs is None:
            continue

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        shape = (args.batch_size, 1, model_kwargs['low_res'].shape[2], model_kwargs['low_res'].shape[3], model_kwargs['low_res'].shape[4])
        logger.log(f"Model input shape: {shape}")  # Debug: 驗證模型輸入維度

        if device == "cuda":
            th.cuda.manual_seed_all(10)
        else:
            th.manual_seed(10)
        noise = th.randn(*shape, device=device)

        if args.use_ddim:
            sample_fn = diffusion.ddim_sample_loop
            extra_args = dict(eta=args.eta)
            logger.log(f"Using DDIM with eta={args.eta}")  # Debug: 確認使用DDIM
        else:
            sample_fn = diffusion.p_sample_loop
            extra_args = {}
            logger.log("Using standard DDPM sampling")

        sample = sample_fn(
            model,
            shape,
            noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            **extra_args,
        )

        logger.log(f"Sample output shape before permute: {sample.shape}")  # Debug: 檢查模型輸出
        sample = sample.permute(0, 1, 3, 4, 2)
        sample = sample.contiguous()
        logger.log(f"Sample output shape after permute: {sample.shape}")  # Debug: 檢查permute後

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_samples_dist = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples_dist, sample)
            for s in all_samples_dist:
                all_images.append(s.cpu().numpy())
        else:
            all_images.append(sample.cpu().numpy())

        logger.log(f"created {len(all_images)} samples")

    if not all_images:
        logger.log("No samples were generated. Exiting.")
        return

    arr = np.concatenate(all_images, axis=0)
    logger.log(f"Concatenated array shape: {arr.shape}")  # Debug: 檢查合併後shape

    if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
        logger.log("Processing TIFF output...")

        try:
            original_data = tifffile.imread(args.base_samples)
            original_depth, original_height, original_width = original_data.shape
            logger.log(f"Original TIFF shape: {original_depth}x{original_height}x{original_width}")

            # 驗證輸入維度符合預期
            assert original_height == 200, f"Expected height 200, got {original_height}"
            assert original_width == 200, f"Expected width 200, got {original_width}"
            assert 90 <= original_depth <= 130, f"Expected depth 90-130, got {original_depth}"

            arr_result = np.zeros((original_height, original_width, original_depth))
            count_arr = np.zeros_like(arr_result)
            
            resolution = args.large_size
            x_starts = _calculate_xy_starts_fixed(original_height, resolution, num_patches=3)
            y_starts = _calculate_xy_starts_fixed(original_width, resolution, num_patches=3)
            z_starts = _calculate_z_starts_with_overlap(original_depth, resolution)
            
            logger.log(f"X starts: {x_starts}")  # Debug: 顯示分割位置
            logger.log(f"Y starts: {y_starts}")
            logger.log(f"Z starts: {z_starts}")
            
            patch_idx = 0
            total_patches = len(x_starts) * len(y_starts) * len(z_starts)
            arr = arr[:total_patches]  # 確保 arr 數量同 patches 匹配

            for x_start in x_starts:
                for y_start in y_starts:
                    for z_start in z_starts:
                        if patch_idx < len(arr):
                            patch = np.squeeze(arr[patch_idx])  # Use squeeze to handle variable single dimensions, ensure 3D
                            logger.log(f"Patch {patch_idx} shape after squeeze: {patch.shape}")  # Debug: 檢查每個patch
                            
                            if patch.ndim != 3:
                                raise ValueError(f"Patch {patch_idx} has unexpected dimensions: {patch.shape}")
                            
                            x_end = min(x_start + resolution, original_height)
                            y_end = min(y_start + resolution, original_width)
                            z_end = min(z_start + resolution, original_depth)
                            
                            hx = x_end - x_start
                            wy = y_end - y_start
                            dz = z_end - z_start
                            
                            logger.log(f"Patch {patch_idx}: ({x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end}) -> ({hx}, {wy}, {dz})")
                            
                            patch_slice = patch[0:hx, 0:wy, 0:dz]
                            arr_result[x_start:x_end, y_start:y_end, z_start:z_end] += patch_slice
                            count_arr[x_start:x_end, y_start:y_end, z_start:z_end] += 1
                            patch_idx += 1
            
            arr_result = np.divide(arr_result, count_arr, where=count_arr != 0)
            overlap_regions = np.sum(count_arr > 1)
            logger.log(f"Reconstruction complete: final shape {arr_result.shape}, overlapped pixels: {overlap_regions}")

        except Exception as e:
            logger.log(f"Reconstruction failed: {e}")
            if len(arr) > 0:
                arr_result = np.squeeze(arr[0])
                if arr_result.ndim != 3:
                    arr_result = np.zeros((args.large_size, args.large_size, args.large_size))  # Fallback to 3D zero array if squeeze fails
            else:
                arr_result = np.zeros((args.large_size, args.large_size, args.large_size))
    else:
        logger.log("Processing NPZ output...")
        if len(arr) > 0:
            arr_result = np.squeeze(arr[0])
            if arr_result.ndim != 3:
                arr_result = np.zeros((args.large_size, args.large_size, args.large_size))  # Fallback to 3D zero array
        else:
            arr_result = np.zeros((args.large_size, args.large_size, args.large_size))

    if not dist.is_initialized() or dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{os.path.basename(args.base_samples).replace('.tif', '')}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr_result)
        
        if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
            tiff_out_path = out_path.replace('.npz', '.tif')
            if arr_result.ndim == 3:
                logger.log(f"Final result shape before transpose: {arr_result.shape}")  # Debug: transpose前
                tiff_data = arr_result.transpose(2, 0, 1)
                logger.log(f"TIFF data shape after transpose: {tiff_data.shape}")  # Debug: transpose後
            else:
                logger.log("Skipping TIFF save due to incorrect dimensions")
                return
            tifffile.imwrite(tiff_out_path, tiff_data.astype(np.float32))
            logger.log(f"Also saved as TIFF: {tiff_out_path}")

    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")

def load_data_for_worker(base_samples, batch_size, class_cond, resolution):
    if not base_samples.endswith(('.tif', '.tiff')):
        logger.log("Unsupported file type")
        yield None
        return

    vol = tifffile.imread(base_samples)
    logger.log(f"Loaded volume with shape: {vol.shape}")  # Debug: 顯示原始載入shape
    
    if vol.ndim == 3:
        D, H, W = vol.shape
        logger.log(f"3D volume: D={D}, H={H}, W={W}")
    elif vol.ndim == 4 and vol.shape[0] >= 2:
        _, D, H, W = vol.shape
        logger.log(f"4D volume: channels={vol.shape[0]}, D={D}, H={H}, W={W}")
        if vol.shape[0] == 1:
            vol = vol[0]  # Reduce to 3D if single channel
            logger.log("Reduced to 3D (single channel)")
        else:
            raise ValueError("Multi-channel TIFF not supported; expected single channel")
    else:
        logger.log("Unsupported TIFF format")
        yield None
        return

    # 驗證維度符合預期
    assert H == 200 and W == 200, f"Expected 200x200 XY dimensions, got {H}x{W}"
    assert 90 <= D <= 130, f"Expected Z dimension 90-130, got {D}"

    # Remove D < resolution check to handle small Z; keep for XY if too small
    if H < resolution or W < resolution:
        logger.log(f"XY too small ({H}x{W}), skipping")
        yield None
        return

    # Handle small Z by padding individual patches (no skip)
    if D < resolution:
        logger.log(f"Z too small ({D}), will pad patches to {resolution}")

    vol[vol > 4] = 4
    vol = vol / 4.0

    x_starts = _calculate_xy_starts_fixed(H, resolution, num_patches=3)
    y_starts = _calculate_xy_starts_fixed(W, resolution, num_patches=3)
    z_starts = _calculate_z_starts_with_overlap(D, resolution)

    total_patches = len(x_starts) * len(y_starts) * len(z_starts)
    logger.log(f"Total expected patches: {total_patches} (X: {len(x_starts)}, Y: {len(y_starts)}, Z: {len(z_starts)})")
    logger.log(f"Patch positions - X: {x_starts}, Y: {y_starts}, Z: {z_starts}")

    image_arr = []
    for x_start in x_starts:
        for y_start in y_starts:
            for z_start in z_starts:
                x_end = min(x_start + resolution, H)
                y_end = min(y_start + resolution, W)
                z_end = min(z_start + resolution, D)
                patch = vol[z_start:z_end, x_start:x_end, y_start:y_end]
                logger.log(f"Raw patch shape: {patch.shape} from vol[{z_start}:{z_end}, {x_start}:{x_end}, {y_start}:{y_end}]")
                
                padded_patch = np.zeros((resolution, resolution, resolution))
                # Pad dimensions: assuming padded_patch as (dz_pad, hx_pad, wy_pad) -> but actually (Z_patch, H_patch, W_patch)
                # patch.shape = (dz, hx, wy)
                dz_actual = patch.shape[0]
                hx_actual = patch.shape[1]
                wy_actual = patch.shape[2]
                padded_patch[:dz_actual, :hx_actual, :wy_actual] = patch
                
                # Debug: 檢查transpose前後
                logger.log(f"Padded patch shape before transpose: {padded_patch.shape}")
                transposed_patch = padded_patch.transpose(1, 2, 0)  # (Z,H,W) -> (H,W,Z) for model
                logger.log(f"Transposed patch shape: {transposed_patch.shape}")
                
                image_arr.append(transposed_patch)

    image_arr = np.array(image_arr)
    logger.log(f"Image array shape: {image_arr.shape}")

    if dist.is_initialized():
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
    else:
        rank = 0
        num_ranks = 1

    # 只處理屬於此 rank 的 patches
    for i in range(rank, len(image_arr), num_ranks):
        batch_patches = [image_arr[i]]
        batch = th.from_numpy(np.stack(batch_patches)).float().permute(0, 3, 1, 2).unsqueeze(1)
        logger.log(f"Batch shape: {batch.shape}")  # Debug: 檢查batch shape
        yield dict(low_res=batch)

def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        eta=0.0,  # Added for DDIM eta control
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def _calculate_xy_starts_fixed(dim_size, patch_size, num_patches=3):
    """固定分割成指定數量的patch (預設3個)，適用於200x200圖像"""
    logger.log(f"Calculating XY starts for dim_size={dim_size}, patch_size={patch_size}, num_patches={num_patches}")
    
    if num_patches == 1:
        return [0]
    
    # 計算每個patch的起始位置，確保有overlap且完全覆蓋
    # 對於200x200，patch_size=96，3個patches的理想分佈
    if dim_size == 200 and patch_size == 96 and num_patches == 3:
        # 手動優化的位置，確保良好的overlap
        starts = [0, 52, 104]  # overlap: 0-96, 52-148, 104-200
        logger.log(f"Using optimized starts for 200x200: {starts}")
        return starts
    
    # 通用計算方法
    total_coverage = dim_size
    if num_patches == 1:
        return [0]
    
    # 計算step，確保最後一個patch能完全覆蓋到邊界
    step = (total_coverage - patch_size) / (num_patches - 1)
    starts = [int(i * step) for i in range(num_patches)]
    
    # 確保最後一個patch不超出邊界
    starts[-1] = min(starts[-1], dim_size - patch_size)
    
    logger.log(f"Calculated starts: {starts}")
    return starts

def _calculate_z_starts_with_overlap(dim_size, patch_size):
    """Z軸處理，適用於連續Z值範圍"""
    logger.log(f"Calculating Z starts for dim_size={dim_size}, patch_size={patch_size}")
    
    if dim_size <= patch_size:
        logger.log(f"Single patch with padding (Z={dim_size} <= {patch_size})")
        return [0]
    
    # 雙patch策略：[0, dim_size - patch_size]
    starts = [0, dim_size - patch_size]
    overlap = patch_size - (dim_size - patch_size)
    overlap_pct = (overlap / patch_size) * 100
    
    logger.log(f"Dual patches: [0:{patch_size}], [{starts[1]}:{dim_size}], overlap={overlap}px ({overlap_pct:.1f}%)")
    
    return starts

if __name__ == "__main__":
    main()