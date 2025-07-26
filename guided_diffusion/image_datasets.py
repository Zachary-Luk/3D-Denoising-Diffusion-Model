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
    def __init__(self, resolution, image_paths, classes=None,
                shard=0, num_shards=1, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.use_non_overlapping = True  # 直接在這裡設置為 True
        
        if self.use_non_overlapping:
            # 預先計算所有不重疊patch嘅位置
            self._prepare_non_overlapping_patches()
        
    def _prepare_non_overlapping_patches(self):
        """固定分割策略：XY軸3段有重疊，Z軸2段，重疊不超過80%"""
        self.patch_info = []  # 每個元素：(file_idx, x_start, y_start, z_start)
        self.volume_info = {}
        
        for file_idx, path in enumerate(self.local_images):
            ext = os.path.splitext(path)[1].lower()
            if ext in (".tif", ".tiff"):
                try:
                    # 讀取volume獲取尺寸
                    vol = tifffile.imread(path)
                    if vol.ndim == 3:  # (D,H,W)
                        D, H, W = vol.shape
                    elif vol.ndim == 4 and vol.shape[0] >= 2:  # (C,D,H,W)
                        _, D, H, W = vol.shape
                    else:
                        continue
                    
                    # 轉置後嘅尺寸：(H,W,D)
                    self.volume_info[file_idx] = (H, W, D)
                    
                    # 檢查volume是否足夠大
                    if H < self.resolution or W < self.resolution or D < self.resolution:
                        print(f"Warning: Volume {path} 太細 ({H}x{W}x{D}), 跳過")
                        continue
                    
                    # 計算分割點，帶80%重疊保護
                    x_starts = self._calculate_xy_starts(H)
                    y_starts = self._calculate_xy_starts(W)
                    z_starts = self._calculate_z_starts(D)
                    
                    # 生成所有patch組合
                    for x_start in x_starts:
                        for y_start in y_starts:
                            for z_start in z_starts:
                                self.patch_info.append((file_idx, x_start, y_start, z_start))
                                
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue

    def _calculate_xy_starts(self, dim_size):
        """計算XY軸起始點，帶80%重疊保護"""
        patch_size = self.resolution  # 通常是80
        overlap = 20  # 固定20 voxel重疊
        stride = patch_size - overlap  # 80-20=60
        max_overlap = int(patch_size * 0.8)  # 80%重疊閾值 = 64 voxels
        
        starts = []
        
        # 第一個patch: 0-80
        starts.append(0)
        
        # 後續patches: 60-140, 120-200, ...
        pos = stride  # 60
        while pos + patch_size <= dim_size:
            # 檢查與前一個patch嘅重疊
            if starts:
                prev_end = starts[-1] + patch_size
                current_start = pos
                overlap_size = max(0, prev_end - current_start)
                
                if overlap_size > max_overlap:
                    pos += stride
                    continue
                    
            starts.append(pos)
            pos += stride
        
        # 如果最後一個patch沒有完全覆蓋到邊界，檢查是否可以添加
        if starts:
            last_end = starts[-1] + patch_size
            if last_end < dim_size:
                # 添加從末尾倒推嘅patch
                last_start = dim_size - patch_size
                if last_start > starts[-1]:  # 避免重複
                    # 檢查重疊
                    prev_end = starts[-1] + patch_size
                    overlap_size = max(0, prev_end - last_start)
                    
                    if overlap_size <= max_overlap:
                        starts.append(last_start)
                        
        return starts

    def _calculate_z_starts(self, dim_size):
        """計算Z軸起始點，帶80%重疊保護"""
        patch_size = self.resolution  # 通常是80
        max_overlap = int(patch_size * 0.8)  # 80%重疊閾值 = 64 voxels
        
        starts = [0]  # 第一個patch: 0-80
        
        if dim_size > patch_size:
            # 第二個patch從末尾倒推: (D-80)-D
            second_start = dim_size - patch_size
            if second_start > 0:  # 確保不重複
                # 檢查重疊
                first_end = 0 + patch_size  # 第一個patch結束位置
                overlap_size = max(0, first_end - second_start)
                
                if overlap_size <= max_overlap:
                    starts.append(second_start)
                    
        return starts

    def __len__(self):
        if self.use_non_overlapping:
            return len(self.patch_info)
        else:
            return len(self.local_images)

    def __getitem__(self, idx):
        if self.use_non_overlapping:
            return self._get_non_overlapping_patch(idx)
        else:
            return self._get_random_patch(idx)
    
    def _get_non_overlapping_patch(self, idx):
        """獲取重疊分割嘅patch（無zero padding）"""
        file_idx, x_start, y_start, z_start = self.patch_info[idx]
        path = self.local_images[file_idx]
        ext = os.path.splitext(path)[1].lower()

        # -------- 1. 讀取數據 --------
        if ext in (".tif", ".tiff"):
            vol = tifffile.imread(path)
            if vol.ndim == 3:  # (D,H,W)
                low_vol = vol
                high_vol = vol
            elif vol.ndim == 4 and vol.shape[0] >= 2:  # (C,D,H,W)
                low_vol, high_vol = vol[0], vol[1]
            else:
                raise ValueError(f"Unsupported TIFF shape {vol.shape}")
            low_vol = low_vol.transpose(1, 2, 0) / 4   # (H,W,D)
            high_vol = high_vol.transpose(1, 2, 0) / 4
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # -------- 2. 提取patch（保證完整尺寸） --------
        H, W, D = low_vol.shape
        
        # 計算結束位置
        x_end = x_start + self.resolution
        y_end = y_start + self.resolution
        z_end = z_start + self.resolution
        
        # 確保不超出邊界（理論上應該不會，因為我哋已經計算好）
        x_end = min(x_end, H)
        y_end = min(y_end, W)
        z_end = min(z_end, D)
        
        low_patch = low_vol[x_start:x_end, y_start:y_end, z_start:z_end]
        high_patch = high_vol[x_start:x_end, y_start:y_end, z_start:z_end]

        # -------- 3. 驗證patch尺寸（應該係完整嘅） --------
        expected_shape = (self.resolution, self.resolution, self.resolution)
        if low_patch.shape != expected_shape:
            # 如果真的有問題，用padding補救（但這應該很少發生）
            pad_low = np.zeros(expected_shape, dtype=np.float32)
            pad_high = np.zeros(expected_shape, dtype=np.float32)
            
            h, w, d = low_patch.shape
            pad_low[:h, :w, :d] = low_patch
            pad_high[:h, :w, :d] = high_patch
            
            low_patch = pad_low
            high_patch = pad_high

        # -------- 4. 打包 --------
        low_np = np.transpose(low_patch[None, ...], (0, 3, 1, 2))  # (1,T,H,W)
        high_np = np.transpose(high_patch[None, ...], (0, 3, 1, 2))

        model_kwargs = {"low_res": low_np}
        if self.local_classes is not None:
            model_kwargs["y"] = np.array(self.local_classes[file_idx], dtype=np.int64)

        return high_np, model_kwargs
    
    def _get_random_patch(self, idx):
        """原來嘅隨機裁剪邏輯（oversampling）"""
        path = self.local_images[idx]
        ext = os.path.splitext(path)[1].lower()

        # -------- 1. 只處理 tiff --------
        if ext in (".tif", ".tiff"):
            vol = tifffile.imread(path)
            if vol.ndim == 3:  # (D,H,W)
                low_vol = vol
                high_vol = vol
            elif vol.ndim == 4 and vol.shape[0] >= 2:  # (C,D,H,W)
                low_vol, high_vol = vol[0], vol[1]
            else:
                raise ValueError(f"Unsupported TIFF shape {vol.shape}")
            low_vol = low_vol.transpose(1, 2, 0) / 4
            high_vol = high_vol.transpose(1, 2, 0) / 4
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # -------- 2. 隨機裁剪 --------
        H, W, D = low_vol.shape
        size_xy = min(self.resolution, H, W)
        size_z = min(self.resolution, D)

        rand_x = np.random.randint(0, max(H-size_xy, 0)+1)
        rand_y = np.random.randint(0, max(W-size_xy, 0)+1)
        rand_z = np.random.randint(0, max(D-size_z, 0)+1)

        low_patch = low_vol[rand_x:rand_x+size_xy,
                           rand_y:rand_y+size_xy,
                           rand_z:rand_z+size_z]
        high_patch = high_vol[rand_x:rand_x+size_xy,
                             rand_y:rand_y+size_xy,
                             rand_z:rand_z+size_z]

        if low_patch.shape != (size_xy, size_xy, size_z):
            pad_low = np.zeros((size_xy, size_xy, size_z), np.float32)
            pad_high = np.zeros_like(pad_low)
            h, w, d = low_patch.shape
            pad_low[:h, :w, :d] = low_patch
            pad_high[:h, :w, :d] = high_patch
            low_patch, high_patch = pad_low, pad_high

        # -------- 3. 打包 --------
        low_np = np.transpose(low_patch[None, ...], (0, 3, 1, 2))  # (1,T,H,W)
        high_np = np.transpose(high_patch[None, ...], (0, 3, 1, 2))

        model_kwargs = {"low_res": low_np}
        if self.local_classes is not None:
            model_kwargs["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return high_np, model_kwargs

    def get_overlap_stats(self):
        """顯示重疊分割統計信息"""
        if not self.use_non_overlapping:
            print("只在 use_non_overlapping=True 時可用")
            return
            
        print(f"重疊分割模式 (patch size: {self.resolution}x{self.resolution}x{self.resolution}):")
        print(f"重疊保護: 最大允許重疊 {int(self.resolution * 0.8)} voxels (80%)")
        
        for file_idx, (H, W, D) in self.volume_info.items():
            x_starts = self._calculate_xy_starts(H)
            y_starts = self._calculate_xy_starts(W)
            z_starts = self._calculate_z_starts(D)
            
            total_patches = len(x_starts) * len(y_starts) * len(z_starts)
            
            print(f"Volume {file_idx} ({H}x{W}x{D}): {total_patches} patches")
            print(f"  X軸: {len(x_starts)} patches at {x_starts}")
            print(f"  Y軸: {len(y_starts)} patches at {y_starts}")
            print(f"  Z軸: {len(z_starts)} patches at {z_starts}")
            
            # 計算實際重疊
            if len(x_starts) > 1:
                x_overlaps = []
                for i in range(len(x_starts)-1):
                    overlap = (x_starts[i] + self.resolution) - x_starts[i+1]
                    x_overlaps.append(overlap)
                print(f"  X軸重疊: {x_overlaps} voxels")
                
            if len(z_starts) > 1:
                z_overlap = (z_starts[0] + self.resolution) - z_starts[1]
                print(f"  Z軸重疊: {z_overlap} voxels")

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

