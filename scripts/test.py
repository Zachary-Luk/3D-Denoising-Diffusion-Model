import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tifffile
import numpy as np
import torch as th
import torch.distributed as dist

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

    dev = dist_util.dev()
    
    # 固定種子（同單 patch 版本一致）
    if dev.type == "cuda":
        th.cuda.manual_seed_all(10)
    else:
        th.manual_seed(10)
    logger.log("Fixed seed set to 10")

    for model_kwargs in data_generator:
        if model_kwargs is None:
            continue

        model_kwargs = {k: v.to(dev) for k, v in model_kwargs.items()}

        # DDPM 采樣（同單 patch 邏輯）
        shape = model_kwargs['low_res'].shape  # (B, 1, Z, H, W)
        logger.log(f"Processing patch with shape={shape}")

        with th.no_grad():
            noise = th.randn(*shape, device=dev)
            sample = diffusion.p_sample_loop(
                model,
                shape,
                noise,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        # 維度調整配合重建：(B,1,Z,H,W) -> (B,1,H,W,Z)
        sample = sample.permute(0, 1, 3, 4, 2).contiguous()

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_samples_dist = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples_dist, sample)
            for s in all_samples_dist:
                all_images.append(s.cpu().numpy())
        else:
            all_images.append(sample.cpu().numpy())

        logger.log(f"Processed patch, total samples: {len(all_images)}")

    if not all_images:
        logger.log("No samples were generated. Exiting.")
        return

    arr = np.concatenate(all_images, axis=0)
    logger.log(f"Concatenated array shape: {arr.shape}")

    # 重建完整圖像（用 Hann window 處理 overlap）
    if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
        logger.log("Reconstructing full image with Hann window blending...")

        try:
            original_data = tifffile.imread(args.base_samples)
            original_depth, original_height, original_width = original_data.shape

            arr_result = np.zeros((original_height, original_width, original_depth), dtype=np.float32)
            weight_arr = np.zeros_like(arr_result, dtype=np.float32)
            
            resolution = args.large_size
            x_starts = _calculate_xy_starts_fixed(original_height, resolution, num_patches=3)
            y_starts = _calculate_xy_starts_fixed(original_width, resolution, num_patches=3)
            z_starts = _calculate_z_starts_with_overlap(original_depth, resolution)
            
            logger.log(f"Reconstruction: X starts: {x_starts}, Y starts: {y_starts}, Z starts: {z_starts}")
            
            # 創建 3D Hann window
            hann_window = create_3d_hann_window(resolution)
            logger.log(f"Created 3D Hann window with shape: {hann_window.shape}")
            
            patch_idx = 0
            total_patches = len(x_starts) * len(y_starts) * len(z_starts)
            arr = arr[:total_patches]

            for x_start in x_starts:
                for y_start in y_starts:
                    for z_start in z_starts:
                        if patch_idx < len(arr):
                            patch = np.squeeze(arr[patch_idx])
                            
                            if patch.ndim != 3:
                                raise ValueError(f"Patch {patch_idx} has unexpected dimensions: {patch.shape}")
                            
                            x_end = min(x_start + resolution, original_height)
                            y_end = min(y_start + resolution, original_width)
                            z_end = min(z_start + resolution, original_depth)
                            
                            hx = x_end - x_start
                            wy = y_end - y_start
                            dz = z_end - z_start
                            
                            logger.log(f"Reconstructing patch {patch_idx}: ({x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end}) -> ({hx}, {wy}, {dz})")
                            
                            # 提取對應尺寸的 patch 和 weight
                            patch_slice = patch[0:hx, 0:wy, 0:dz]
                            weight_slice = hann_window[0:hx, 0:wy, 0:dz]
                            
                            # 用 Hann window 加權累積
                            arr_result[x_start:x_end, y_start:y_end, z_start:z_end] += patch_slice * weight_slice
                            weight_arr[x_start:x_end, y_start:y_end, z_start:z_end] += weight_slice
                            patch_idx += 1
            
            # 正規化：除以總權重
            arr_result = np.divide(arr_result, weight_arr, where=weight_arr > 0)
            
            # 統計
            overlap_regions = np.sum(weight_arr > hann_window.max() * 1.1)  # 超過單個 window 最大值的區域
            logger.log(f"Reconstruction complete with Hann blending: final shape {arr_result.shape}")
            logger.log(f"Overlapped regions (Hann weighted): {overlap_regions}")
            
            # 整體質量評估
            original_std = np.std(original_data.astype(np.float32))
            denoised_std = np.std(arr_result)
            noise_reduction = (original_std - denoised_std) / original_std * 100 if original_std > 0 else 0
            
            logger.log(f"Full image denoising results:")
            logger.log(f"  Original std: {original_std:.4f}")
            logger.log(f"  Denoised std: {denoised_std:.4f}")
            logger.log(f"  Noise reduction: {noise_reduction:.1f}%")

        except Exception as e:
            logger.log(f"Reconstruction failed: {e}")
            arr_result = np.zeros((args.large_size, args.large_size, args.large_size))

    # 保存最終結果
    if not dist.is_initialized() or dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"denoised_{os.path.basename(args.base_samples).replace('.tif', '')}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr_result)
        
        if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
            tiff_out_path = out_path.replace('.npz', '.tif')
            if arr_result.ndim == 3:
                # 直接保存，無額外 scaling
                tiff_data = arr_result.transpose(2, 0, 1)  # (H,W,Z) -> (Z,H,W)
                tifffile.imwrite(tiff_out_path, tiff_data.astype(np.float32))
                logger.log(f"Saved denoised TIFF: {tiff_out_path}")

    if dist.is_initialized():
        dist.barrier()
    logger.log("Full image denoising complete")

def load_data_for_worker(base_samples, batch_size, class_cond, resolution):
    """載入並切割成多個 patches（無 normalization）"""
    if not base_samples.endswith(('.tif', '.tiff')):
        logger.log("Unsupported file type")
        yield None
        return

    vol = tifffile.imread(base_samples)
    
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    
    D, H, W = vol.shape
    assert H == 200 and W == 200, f"Expected 200x200 XY dimensions, got {H}x{W}"
    assert 90 <= D <= 130, f"Expected Z dimension 90-130, got {D}"

    # 直接用原始數據，無 normalization
    vol = vol.astype(np.float32)
    logger.log(f"Using original data without normalization - min: {vol.min():.4f}, max: {vol.max():.4f}, std: {vol.std():.4f}")

    # 計算 patch 位置
    x_starts = _calculate_xy_starts_fixed(H, resolution, num_patches=3)
    y_starts = _calculate_xy_starts_fixed(W, resolution, num_patches=3)
    z_starts = _calculate_z_starts_with_overlap(D, resolution)

    total_patches = len(x_starts) * len(y_starts) * len(z_starts)
    logger.log(f"Total patches to process: {total_patches}")

    # 準備所有 patches
    image_arr = []
    for x_start in x_starts:
        for y_start in y_starts:
            for z_start in z_starts:
                x_end = min(x_start + resolution, H)
                y_end = min(y_start + resolution, W)
                z_end = min(z_start + resolution, D)
                patch = vol[z_start:z_end, x_start:x_end, y_start:y_end]
                
                # Pad 到標準尺寸
                padded_patch = np.zeros((resolution, resolution, resolution), dtype=np.float32)
                dz, hx, wy = patch.shape
                padded_patch[:dz, :hx, :wy] = patch
                
                # 轉為 (H,W,Z) 格式
                transposed_patch = padded_patch.transpose(1, 2, 0)  # (Z,H,W) -> (H,W,Z)
                image_arr.append(transposed_patch)

    image_arr = np.array(image_arr)

    # 分佈式處理
    if dist.is_initialized():
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
    else:
        rank = 0
        num_ranks = 1

    # 逐個 yield patches
    for i in range(rank, len(image_arr), num_ranks):
        batch_patches = [image_arr[i]]
        batch = th.from_numpy(np.stack(batch_patches)).float().permute(0, 3, 1, 2).unsqueeze(1)  # (1,1,Z,H,W)
        yield dict(low_res=batch)

def create_3d_hann_window(size):
    """創建 3D Hann window 用於 smooth blending"""
    # 創建 1D Hann window
    hann_1d = np.hanning(size)
    
    # 擴展到 3D
    hann_3d = np.outer(hann_1d, hann_1d)  # 2D
    hann_3d = np.outer(hann_3d.flatten(), hann_1d).reshape(size, size, size)  # 3D
    
    # 正規化到 [0, 1]
    hann_3d = hann_3d / hann_3d.max()
    
    logger.log(f"Hann window stats - min: {hann_3d.min():.4f}, max: {hann_3d.max():.4f}, mean: {hann_3d.mean():.4f}")
    
    return hann_3d

def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,  # 用 DDPM
        eta=0.0,
        timestep_respacing="",
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def _calculate_xy_starts_fixed(dim_size, patch_size, num_patches=3):
    """固定分割成指定数量的patch"""
    if dim_size == 200 and patch_size == 96 and num_patches == 3:
        return [0, 52, 104]
    
    if num_patches == 1:
        return [0]
    
    step = (dim_size - patch_size) / (num_patches - 1)
    starts = [int(i * step) for i in range(num_patches)]
    starts[-1] = min(starts[-1], dim_size - patch_size)
    return starts

def _calculate_z_starts_with_overlap(dim_size, patch_size):
    """Z轴处理"""
    if dim_size <= patch_size:
        return [0]
    
    starts = [0, dim_size - patch_size]
    return starts

if __name__ == "__main__":
    main()