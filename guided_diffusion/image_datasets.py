import math
import random
import os

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

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
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    except:
        rank = 0
        size = 1
    
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=rank,
        num_shards=size,
        random_crop=random_crop,
        random_flip=random_flip,
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


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npz"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results



# pet and ct data 3D
# normalize to (-1,1)
class ImageDataset(Dataset):
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
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        pet_data = np.load(path)['arr_0']
        # pet_data = pet_data.astype(np.float32) * 2.0 - 1.0
        pet_data = pet_data.astype(np.float32)

        pet_data = pet_data/4 # add nmlz
        # pet_data = pet_data * 2.0 - 1.0

        # input 96*96*96
        # size = 96  # patch size
        # rand_xyz = np.random.randint(0, 144-size+1, 3)
        # pet_low = pet_data[0:2,rand_xyz[0]:rand_xyz[0]+size,rand_xyz[1]:rand_xyz[1]+size,rand_xyz[2]:rand_xyz[2]+size].copy()
        # label = pet_data[2,rand_xyz[0]:rand_xyz[0]+size,rand_xyz[1]:rand_xyz[1]+size,rand_xyz[2]:rand_xyz[2]+size].copy()
        # while label.max() == 0:
        #     rand_xyz = np.random.randint(0, 144-size+1, 3)
        #     pet_low = pet_data[0:2,rand_xyz[0]:rand_xyz[0]+size,rand_xyz[1]:rand_xyz[1]+size,rand_xyz[2]:rand_xyz[2]+size].copy()
        #     label = pet_data[2,rand_xyz[0]:rand_xyz[0]+size,rand_xyz[1]:rand_xyz[1]+size,rand_xyz[2]:rand_xyz[2]+size].copy()
        # C, H, W, T = pet_low.shape

        # input 96*96*32
        size_xy = 96  # patch size
        size_z = 96 #32
        rand_x = np.random.randint(0, 180-size_xy+1)
        rand_y = np.random.randint(0, 280-size_xy+1) 
        rand_z = np.random.randint(0, 520-size_z+1) 
        pet_low = pet_data[0, rand_x:rand_x+size_xy, rand_y:rand_y+size_xy, rand_z:rand_z+size_z].copy()
        label = pet_data[1, rand_x:rand_x+size_xy, rand_y:rand_y+size_xy, rand_z:rand_z+size_z].copy()
        # while label.max() == label.min():
        #     rand_xy = np.random.randint(0, 144-size_xy+1, 2)
        #     rand_z = np.random.randint(0, 144-size_z+1)
        #     pet_low = pet_data[0:2, rand_xy[0]:rand_xy[0]+size_xy, rand_xy[1]:rand_xy[1]+size_xy, rand_z:rand_z+size_z].copy()
        #     label = pet_data[2, rand_xy[0]:rand_xy[0]+size_xy, rand_xy[1]:rand_xy[1]+size_xy, rand_z:rand_z+size_z].copy()
        H, W, T = pet_low.shape

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(pet_low.reshape((1, H, W, T)), [0, 3, 1, 2]), np.transpose(label.reshape((1, H, W, T)), [0, 3, 1, 2]), out_dict



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
