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

    if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
        logger.log("Processing TIFF output...")

        try:
            original_data = tifffile.imread(args.base_samples)
            original_depth, original_height, original_width = original_data.shape
            logger.log(f"Original TIFF shape: {original_depth}x{original_height}x{original_width}")

            arr_result = np.zeros((original_height, original_width, original_depth))
            count_arr = np.zeros_like(arr_result)
            
            resolution = args.large_size
            x_starts = _calculate_xy_starts(original_height, resolution)
            y_starts = _calculate_xy_starts(original_width, resolution)
            z_starts = _calculate_z_starts(original_depth, resolution)
            
            patch_idx = 0
            total_patches = len(x_starts) * len(y_starts) * len(z_starts)
            arr = arr[:total_patches]  # 確保 arr 數量同 patches 匹配

            for x_start in x_starts:
                for y_start in y_starts:
                    for z_start in z_starts:
                        if patch_idx < len(arr):
                            patch = arr[patch_idx]
                            x_end = min(x_start + resolution, original_height)
                            y_end = min(y_start + resolution, original_width)
                            z_end = min(z_start + resolution, original_depth)
                            
                            patch_slice = patch[0:x_end-x_start, 0:y_end-y_start, 0:z_end-z_start]
                            arr_result[x_start:x_end, y_start:y_end, z_start:z_end] += patch_slice
                            count_arr[x_start:x_end, y_start:y_end, z_start:z_end] += 1
                            patch_idx += 1
            
            arr_result = np.divide(arr_result, count_arr, where=count_arr != 0)
            logger.log(f"Reconstruction complete: final shape {arr_result.shape}")

        except Exception as e:
            logger.log(f"Reconstruction failed: {e}")
            arr_result = arr[0] if len(arr) > 0 else np.zeros((args.large_size, args.large_size, args.large_size))
    else:
        logger.log("Processing NPZ output...")
        arr_result = np.zeros((192,288,576))
        index = 0
        for i in range(6):
            arr_result[:, :, i*96:(i+1)*96] = arr[index,:,:,:]
            index += 1

    if not dist.is_initialized() or dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{os.path.basename(args.base_samples).replace('.tif', '')}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr_result)
        
        if args.base_samples.endswith('.tif') or args.base_samples.endswith('.tiff'):
            tiff_out_path = out_path.replace('.npz', '.tif')
            tiff_data = arr_result.transpose(2, 0, 1)
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
    if vol.ndim == 3:
        D, H, W = vol.shape
    elif vol.ndim == 4 and vol.shape[0] >= 2:
        _, D, H, W = vol.shape
    else:
        logger.log("Unsupported TIFF format")
        yield None
        return

    if H < resolution or W < resolution or D < resolution:
        logger.log(f"Volume too small ({H}x{W}x{D}), skipping")
        yield None
        return

    vol[vol > 4] = 4
    vol = vol / 4.0

    x_starts = _calculate_xy_starts(H, resolution)
    y_starts = _calculate_xy_starts(W, resolution)
    z_starts = _calculate_z_starts(D, resolution)

    total_patches = len(x_starts) * len(y_starts) * len(z_starts)
    logger.log(f"Total expected patches: {total_patches}")

    image_arr = []
    for x_start in x_starts:
        for y_start in y_starts:
            for z_start in z_starts:
                x_end = min(x_start + resolution, H)
                y_end = min(y_start + resolution, W)
                z_end = min(z_start + resolution, D)
                patch = vol[z_start:z_end, x_start:x_end, y_start:y_end]
                padded_patch = np.zeros((resolution, resolution, resolution))
                padded_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                image_arr.append(padded_patch.transpose(1, 2, 0))

    image_arr = np.array(image_arr)

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
        yield dict(low_res=batch)

def create_argparser():
    defaults = dict(
        save_dir='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
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
            if max(0, prev_end - pos) > max_overlap:
                pos += stride
                continue
        starts.append(pos)
        pos += stride
    if starts and starts[-1] + patch_size < dim_size:
        last_start = dim_size - patch_size
        if last_start > starts[-1] and max(0, starts[-1] + patch_size - last_start) <= max_overlap:
            starts.append(last_start)
    return starts

def _calculate_z_starts(dim_size, patch_size):
    max_overlap = int(patch_size * 0.8)
    starts = [0]
    if dim_size > patch_size:
        second_start = dim_size - patch_size
        if second_start > 0 and max(0, patch_size - second_start) <= max_overlap:
            starts.append(second_start)
    return starts

if __name__ == "__main__":
    main()