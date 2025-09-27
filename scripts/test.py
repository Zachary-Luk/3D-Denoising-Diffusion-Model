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

    logger.log("loading single patch...")
    patch_data = load_single_patch(args.base_samples, args.large_size)
    
    if patch_data is None:
        logger.log("Failed to load patch. Exiting.")
        return

    dev = dist_util.dev()

    # 準備模型輸入
    low_res_patch = th.from_numpy(patch_data).float().unsqueeze(0).unsqueeze(0).to(dev)  # (1,1,H,W,Z)
    low_res_patch = low_res_patch.permute(0, 1, 4, 2, 3)  # (1,1,Z,H,W) - 模型期望格式
    
    shape = low_res_patch.shape
    model_kwargs = {"low_res": low_res_patch}
    
    logger.log(f"Input patch shape: {shape}")
    logger.log(f"Input stats - min: {low_res_patch.min():.4f}, max: {low_res_patch.max():.4f}, mean: {low_res_patch.mean():.4f}, std: {low_res_patch.std():.4f}")

    # 固定種子同原版 PET 一致
    if dev.type == "cuda":
        th.cuda.manual_seed_all(10)
    else:
        th.manual_seed(10)
    
    # DDPM 采樣（用固定 noise）
    logger.log("Starting DDPM denoising...")
    with th.no_grad():
        noise = th.randn(*shape, device=dev)
        sample = diffusion.p_sample_loop(
            model,
            shape,
            noise,  # 固定 noise
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

    logger.log(f"Output stats - min: {sample.min():.4f}, max: {sample.max():.4f}, mean: {sample.mean():.4f}, std: {sample.std():.4f}")
    
    # 轉回 (Z,H,W) 格式保存
    denoised_patch = sample[0, 0].cpu().numpy()  # (Z,H,W)
    original_patch = low_res_patch[0, 0].cpu().numpy()  # (Z,H,W)
    
    # 保存結果：去噪前後對比
    if not dist.is_initialized() or dist.get_rank() == 0:
        # 保存去噪前 TIFF
        original_tiff = os.path.join(logger.get_dir(), "original_noisy_patch.tif")
        tifffile.imwrite(original_tiff, original_patch.astype(np.float32))
        logger.log(f"Saved original noisy patch: {original_tiff}")
        
        # 保存去噪後 TIFF
        denoised_tiff = os.path.join(logger.get_dir(), "denoised_patch.tif")
        tifffile.imwrite(denoised_tiff, denoised_patch.astype(np.float32))
        logger.log(f"Saved denoised patch: {denoised_tiff}")
        
        # 簡單質量對比
        original_std = original_patch.std()
        denoised_std = denoised_patch.std()
        noise_reduction = (original_std - denoised_std) / original_std * 100 if original_std > 0 else 0
        
        logger.log(f"Denoising results:")
        logger.log(f"  Original std: {original_std:.4f}")
        logger.log(f"  Denoised std: {denoised_std:.4f}")
        logger.log(f"  Noise reduction: {noise_reduction:.1f}%")
        
        if noise_reduction > 10:
            logger.log("✓ Good denoising!")
        elif noise_reduction > 0:
            logger.log("~ Mild denoising")
        else:
            logger.log("✗ No denoising effect")

    logger.log("Denoising complete - check the two TIFF files for comparison")

def load_single_patch(base_samples, resolution):
    """載入並提取中心 patch"""
    if not base_samples.endswith(('.tif', '.tiff')):
        logger.log("Unsupported file type")
        return None

    vol = tifffile.imread(base_samples)
    logger.log(f"Loaded volume with shape: {vol.shape}")
    
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    
    if vol.ndim != 3:
        logger.log("Expected 3D volume")
        return None
        
    D, H, W = vol.shape
    logger.log(f"3D volume: D={D}, H={H}, W={W}")
    
    # 正規化
    logger.log(f"Original stats - min: {vol.min():.4f}, max: {vol.max():.4f}, std: {vol.std():.4f}")
    vol[vol > 4] = 4
    vol = vol / 4.0
    logger.log(f"After normalization - min: {vol.min():.4f}, max: {vol.max():.4f}, std: {vol.std():.4f}")
    
    # 提取中心 patch
    z_center = D // 2
    h_center = H // 2
    w_center = W // 2
    
    z_start = max(0, z_center - resolution // 2)
    z_end = min(D, z_start + resolution)
    h_start = max(0, h_center - resolution // 2)
    h_end = min(H, h_start + resolution)
    w_start = max(0, w_center - resolution // 2)
    w_end = min(W, w_start + resolution)
    
    patch = vol[z_start:z_end, h_start:h_end, w_start:w_end]
    logger.log(f"Extracted patch shape: {patch.shape}")
    
    # Pad 到標準尺寸
    padded_patch = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    dz, hx, wy = patch.shape
    padded_patch[:dz, :hx, :wy] = patch
    
    # 轉為 (H,W,Z) 格式
    result_patch = padded_patch.transpose(1, 2, 0)  # (Z,H,W) -> (H,W,Z)
    logger.log(f"Final patch shape: {result_patch.shape}")
    
    return result_patch

def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        eta=0.0,
        base_samples="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()