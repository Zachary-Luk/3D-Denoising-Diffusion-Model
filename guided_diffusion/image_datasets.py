import math
import random
import os
import tifffile
from PIL import Image
import blobfile as bf
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
import numpy as np
import torch

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    # Handle single GPU vs multi-GPU
    try:
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    except:
        rank = 0
        size = 1
    
    dataset = ImageDataset(            # ← 換成 ImageDataset
            image_size,
            all_files,
            classes=classes,
            shard=rank,
            num_shards=size,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_tif(path):
    """讀取 TIFF 並返回 [C, D, H, W] 格式"""
    try:
        img = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(img)  # [D, H, W] 或 [H, W]
        
        if len(array.shape) == 3:  # 3D
            array = np.expand_dims(array, axis=0)  # [C=1, D, H, W]
        elif len(array.shape) == 2:  # 2D  
            array = np.expand_dims(np.expand_dims(array, axis=0), axis=0)  # [C=1, D=1, H, W]
        
        tensor = torch.from_numpy(array.astype(np.float32))
        return tensor
    except Exception as e:
        raise ValueError(f"Error reading TIFF {path}: {e}")

class CustomImageDataset(Dataset):
    def __init__(
        self,
        resolution,  # image_size
        image_paths,  # all_files
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard::num_shards]  # 分片邏輯
        self.classes = None if classes is None else classes[shard::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
            arr = load_tif(path)  # [C, D, H, W]
        else:
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")
            arr = np.array(pil_image).astype(np.float32)
            arr = arr.transpose(2, 0, 1)  # [C, H, W]
            arr = torch.from_numpy(arr)

        # 構建 model_kwargs
        model_kwargs = {}
        model_kwargs["low_res"] = arr.clone()  # SuperRes 需要的條件
        if self.classes is not None:
            model_kwargs["y"] = torch.tensor(self.classes[idx])
        
        # 返回正確格式：(image_tensor, model_kwargs_dict)
        return arr, model_kwargs


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif", "tiff"]:  # 添加 tif 支持
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results



class ImageDataset(Dataset):
    """
    讀取 3D 醫學影像 (TIFF 或 NPZ) ，隨機裁剪
    (self.resolution × self.resolution × self.resolution) patch。
    傳回：
        low_patch  -> 形狀 (1,  D, H, W)
        high_patch -> 形狀 (1,  D, H, W)
    """
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images  = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    # ------------------- 基本 API -------------------
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # -------- 1. 讀檔並得到 low_vol, high_vol (H,W,D) --------
        if path.endswith((".tif", ".tiff")):
            vol = tifffile.imread(path)                     # (D,H,W) or (C,D,H,W)
            if vol.ndim == 3:                               # (D,H,W) 單通道
                low_vol  = vol
                high_vol = vol
            elif vol.ndim == 4 and vol.shape[0] >= 2:       # (C,D,H,W)
                low_vol  = vol[0]
                high_vol = vol[1]
            else:
                raise ValueError(f"Unsupported TIFF shape {vol.shape}")
            low_vol  = low_vol .transpose(1, 2, 0) / 4.0    # -> (H,W,D)
            high_vol = high_vol.transpose(1, 2, 0) / 4.0
        # else:                                               # NPZ 分支
        #     vol = np.load(path)["arr_0"].astype(np.float32) / 4.0  # (2,H,W,D)
        #     low_vol  = vol[0]
        #     high_vol = vol[1]

        # -------- 2. 隨機裁剪 patch --------
        H, W, D   = low_vol.shape
        size_xy   = min(self.resolution, H, W)
        size_z    = min(self.resolution, D)

        max_x = max(H - size_xy, 0)
        max_y = max(W - size_xy, 0)
        max_z = max(D - size_z , 0)

        rand_x = np.random.randint(0, max_x + 1)
        rand_y = np.random.randint(0, max_y + 1)
        rand_z = np.random.randint(0, max_z + 1)

        low_patch  = low_vol [rand_x:rand_x+size_xy,
                              rand_y:rand_y+size_xy,
                              rand_z:rand_z+size_z]
        high_patch = high_vol[rand_x:rand_x+size_xy,
                              rand_y:rand_y+size_xy,
                              rand_z:rand_z+size_z]

        # -------- 3. Pad 不足尺寸 --------
        if low_patch.shape != (size_xy, size_xy, size_z):
            pad_shape = (size_xy, size_xy, size_z)
            pad_low  = np.zeros(pad_shape, dtype=np.float32)
            pad_high = np.zeros(pad_shape, dtype=np.float32)
            h, w, d  = low_patch.shape
            pad_low [:h, :w, :d] = low_patch
            pad_high[:h, :w, :d] = high_patch
            low_patch , high_patch = pad_low, pad_high

        # -------- 4. 轉成 (C=1, D, H, W) --------
        C, H, W, T = 1, *low_patch.shape
        low_np  = np.transpose(low_patch .reshape((C, H, W, T)), (0, 3, 1, 2))
        high_np = np.transpose(high_patch.reshape((C, H, W, T)), (0, 3, 1, 2))

        model_kwargs = {"low_res": low_np}
        if self.local_classes is not None:
            model_kwargs["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return high_np, model_kwargs      # 只回傳 2-tuple

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


























    # class ImageDataset(Dataset):
#     def __init__(
#         self,
#         resolution,
#         image_paths,
#         classes=None,
#         shard=0,
#         num_shards=1,
#         random_crop=False,
#         random_flip=False,
#     ):
#         super().__init__()
#         self.resolution = resolution
#         self.local_images = image_paths[shard:][::num_shards]
#         self.local_classes = None if classes is None else classes[shard:][::num_shards]
#         self.random_crop = random_crop
#         self.random_flip = random_flip

#     def __len__(self):
#         return len(self.local_images)

#     def __getitem__(self, idx):
#         path = self.local_images[idx]

#         # 檢查文件類型並相應處理
#         if path.endswith('.tif') or path.endswith('.tiff'):
#             pet_data = tifffile.imread(path)  # Shape: (depth, height, width) 或其他
            
#             # 根據你嘅 TIFF 數據結構調整
#             # 假設你嘅 TIFF 係 (depth, height, width) 格式
#             if len(pet_data.shape) == 3:
#                 # 如果係單通道 3D 數據，需要創建 low-dose 同 high-dose 對
#                 # 你可能需要根據實際情況調整呢個邏輯
                
#                 # 選項 1: 如果你有成對嘅文件（low-dose 同 high-dose）
#                 # 你需要修改文件加載邏輯嚟同時加載兩個文件
                
#                 # 選項 2: 如果單一文件包含兩個通道
#                 # pet_data 應該係 (2, depth, height, width) 或類似格式
                
#                 # 暫時假設你會提供成對數據，呢度用同一數據作為示例
#                 pet_low_data = pet_data.copy()  # 低劑量數據
#                 pet_high_data = pet_data.copy()  # 高劑量數據（實際應該係不同文件）
                
#                 # 重新排列維度適應現有邏輯: (height, width, depth)
#                 pet_low_data = pet_low_data.transpose(1, 2, 0)  # (D,H,W) -> (H,W,D)
#                 pet_high_data = pet_high_data.transpose(1, 2, 0)
                
#             elif len(pet_data.shape) == 4:
#                 # 如果已經係 (channels, depth, height, width) 或類似
#                 pet_low_data = pet_data[0].transpose(1, 2, 0)  # 第一個通道
#                 pet_high_data = pet_data[1].transpose(1, 2, 0)  # 第二個通道
#             else:
#                 raise ValueError(f"Unsupported TIFF shape: {pet_data.shape}")
                
#             # 數據正規化
#             pet_low_data = pet_low_data.astype(np.float32) / 4
#             pet_high_data = pet_high_data.astype(np.float32) / 4
            
#             # 獲取實際尺寸
#             H, W, D = pet_low_data.shape  # 例如 (200, 200, 105)
            
#             # 調整隨機裁剪邏輯適應你嘅數據尺寸
#             size_xy = min(96, min(H, W))  # 確保唔會超出邊界
#             size_z = min(96, D)
            
#             # 隨機裁剪
#             if H > size_xy:
#                 rand_x = np.random.randint(0, H - size_xy + 1)
#             else:
#                 rand_x = 0
                
#             if W > size_xy:
#                 rand_y = np.random.randint(0, W - size_xy + 1)
#             else:
#                 rand_y = 0
                
#             if D > size_z:
#                 rand_z = np.random.randint(0, D - size_z + 1)
#             else:
#                 rand_z = 0
            
#             # 提取 patch
#             pet_low = pet_low_data[rand_x:rand_x+size_xy, rand_y:rand_y+size_xy, rand_z:rand_z+size_z].copy()
#             label = pet_high_data[rand_x:rand_x+size_xy, rand_y:rand_y+size_xy, rand_z:rand_z+size_z].copy()
            
#             # 如果裁剪後尺寸唔夠，進行 padding
#             if pet_low.shape[0] < size_xy or pet_low.shape[1] < size_xy or pet_low.shape[2] < size_z:
#                 padded_low = np.zeros((size_xy, size_xy, size_z), dtype=np.float32)
#                 padded_label = np.zeros((size_xy, size_xy, size_z), dtype=np.float32)
                
#                 actual_h, actual_w, actual_d = pet_low.shape
#                 padded_low[:actual_h, :actual_w, :actual_d] = pet_low
#                 padded_label[:actual_h, :actual_w, :actual_d] = label
                
#                 pet_low = padded_low
#                 label = padded_label
            
#         else:
#             # 原來 NPZ 處理邏輯
#             pet_data = np.load(path)['arr_0']
#             pet_data = pet_data.astype(np.float32)
#             pet_data = pet_data/4

#             # 原來嘅隨機裁剪邏輯
#             size_xy = 96
#             size_z = 96
#             rand_x = np.random.randint(0, 180-size_xy+1)
#             rand_y = np.random.randint(0, 280-size_xy+1) 
#             rand_z = np.random.randint(0, 520-size_z+1) 
#             pet_low = pet_data[0, rand_x:rand_x+size_xy, rand_y:rand_y+size_xy, rand_z:rand_z+size_z].copy()
#             label = pet_data[1, rand_x:rand_x+size_xy, rand_y:rand_y+size_xy, rand_z:rand_z+size_z].copy()

#         H, W, T = pet_low.shape

#         out_dict = {}
#         if self.local_classes is not None:
#             out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
#         return np.transpose(pet_low.reshape((1, H, W, T)), [0, 3, 1, 2]), np.transpose(label.reshape((1, H, W, T)), [0, 3, 1, 2]), out_dict

