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

    # Only barrier if distributed is initialized
    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    if base_samples.endswith('.tif') or base_samples.endswith('.tiff'):
        # 讀取 TIFF 文件

        # 讀取 3D TIFF
        low_pet = tifffile.imread(base_samples)  # Shape: (depth, height, width)
        
        # 或者用 PIL 方法（如果 tifffile 有問題）
        # from PIL import Image
        # img = Image.open(base_samples)
        # frames = []
        # try:
        #     while True:
        #         frames.append(np.array(img))
        #         img.seek(img.tell() + 1)
        # except EOFError:
        #     pass
        # low_pet = np.stack(frames, axis=0)
        
        # 獲取實際尺寸
        depth, height, width = low_pet.shape  # 例如 (105, 200, 200)
        
        # 根據你嘅數據尺寸調整 patch 策略
        if depth <= 96:
            # 如果深度小於等於 96，用一個 patch，需要 padding
            image_arr = np.zeros((1, height, width, 96))
            pad_before = (96 - depth) // 2
            pad_after = 96 - depth - pad_before
            padded_volume = np.pad(low_pet, ((pad_before, pad_after), (0, 0), (0, 0)), 
                                  mode='constant', constant_values=0)
            image_arr[0, :, :, :] = padded_volume.transpose(1, 2, 0)  # (D,H,W) -> (H,W,D)
        else:
            # 如果深度大於 96，分成多個 patches（類似原來邏輯）
            # 你可以根據實際情況調整
            num_patches = 2  # 或者動態計算
            patch_size = 64  # 調整 patch 大小
            image_arr = np.zeros((num_patches, height, width, patch_size))
            
            # 簡單分割（你可以根據需要調整重疊策略）
            if depth == 105:
                # 分成 2 個重疊 patches
                image_arr[0, :, :, :] = low_pet[0:64, :, :].transpose(1, 2, 0)
                image_arr[1, :, :, :] = low_pet[41:105, :, :].transpose(1, 2, 0)
            elif depth == 102:
                image_arr[0, :, :, :] = low_pet[0:64, :, :].transpose(1, 2, 0)
                image_arr[1, :, :, :] = low_pet[38:102, :, :].transpose(1, 2, 0)
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
