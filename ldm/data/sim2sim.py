import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import imageio

from turbojpeg import TurboJPEG
from turbojpeg import TJPF_RGB

from collections import defaultdict
import numpy as np
import threading
import time

timing_stats = defaultdict(list)
timing_lock = threading.Lock()  # 添加锁以确保线程安全
batch_counter = 0
print_interval = 10  # 每10个批次打印一次统计结果

def timing_decorator(func_name):
    """函数计时装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            with timing_lock:
                timing_stats[func_name].append(elapsed)
            return result
        return wrapper
    return decorator


class TimingContext:
    """上下文管理器用于代码块计时"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        with timing_lock:
            timing_stats[self.name].append(elapsed)


def print_timing_stats():
    """打印时间统计信息"""
    print("\n===== 数据加载时间统计 =====")
    for operation, times in sorted(timing_stats.items()):
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"{operation:<30} - 平均: {avg_time*1000:.2f}ms, 最大: {max_time*1000:.2f}ms, 最小: {min_time*1000:.2f}ms, 调用次数: {len(times)}")
    print("=========================\n")


class Sim2SimEmpty(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return int(1e6)

    def __getitem__(self, i):
        example = {
            "canonical_image": np.random.rand(256, 256, 3),
            "random_image": np.random.rand(256, 256, 3),
        }
        return example


class Sim2SimBase(Dataset):
    def __init__(self,
                 len_file,
                 data_root,
                 split,
                 original_size=(240, 320),
                 out_size=(256, 256),
                 crop_ratio=0.95,
                 flip_p=0.5,
                 interpolation="bicubic"
                 ):
        with TimingContext("dataset_initialization"):
            self.jpeg = TurboJPEG()

            self.data_root = data_root

            with open(len_file, "rb") as f:
                self.len_file = pickle.load(f)

            self.len_file = [(length - 1) // 10 + 1 if length > 0 else 0 for length in self.len_file]

            train_len = int(len(self.len_file) * 0.99)

            if split == "train":
                self.len_file = self.len_file[:train_len]
                self.start_len = 0
            else:
                self.len_file = self.len_file[train_len:]
                self.start_len = train_len

            self.cum_len = np.cumsum(self.len_file)

            self.out_size = out_size
            self.interpolation = {
                "bilinear": PIL.Image.BILINEAR,
                "bicubic": PIL.Image.BICUBIC,
                "lanczos": PIL.Image.LANCZOS,
            }[interpolation]

            self.transforms = [
                transforms.RandomCrop(size=[int(original_size[0] * crop_ratio), int(original_size[1] * crop_ratio)]),
                transforms.Resize(out_size, interpolation=self.interpolation),
                # transforms.RandomRotation(degrees=[-15.0, 15.0], expand=False),
                # transforms.RandomHorizontalFlip(p=flip_p),
            ]

            self.rand_only_transforms = [
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
            ]

            with TimingContext("get item example"):
                example = self.__getitem__(0)

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, index):
        file_index = np.argmax(self.cum_len > index)
        if file_index:
            step = index - self.cum_len[file_index - 1]
        else:
            step = index

        with TimingContext("load_image"):
            image_0_path = os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"env_rand_cam_third_step_{step * 10}.jpg")
            image_1_path = os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"env_cano_cam_third_step_{step * 10}.jpg")

            with open(image_0_path, 'rb') as f0, open(image_1_path, 'rb') as f1:
                jpeg_1 = f1.read()
                jpeg_0 = f0.read()
                image_0 = self.jpeg.decode(jpeg_0, pixel_format=TJPF_RGB)  #0
                image_1 = self.jpeg.decode(jpeg_1, pixel_format=TJPF_RGB)  #0

        with TimingContext("image process"):
            image_0 = np.array(image_0)
            image_1 = np.array(image_1)

            images = np.stack((image_0, image_1), axis=0)
            images = torch.from_numpy(images)  # (2, h, w, c)
            images = torch.einsum('k h w c -> k c h w', images)

            for transform in self.transforms:
                images = transform(images)
            
            image_0, image_1 = images[0], images[1]

            for transform in self.rand_only_transforms:
                image_0 = transform(image_0)

            image_0 = torch.einsum('c h w -> h w c', image_0)
            image_1 = torch.einsum('c h w -> h w c', image_1)

            image_0 = (image_0 / 127.5 - 1.0).float().numpy()
            image_1 = (image_1 / 127.5 - 1.0).float().numpy()

            example = dict(
                canonical_image=image_1,
                random_image=image_0
            )

        return example

class Sim2SimBasePre(Dataset):
    def __init__(self,
                 len_file,
                 data_root,
                 split,
                 ):
        with TimingContext("dataset_initialization"):
            self.jpeg = TurboJPEG()

            self.data_root = data_root

            with open(len_file, "rb") as f:
                self.len_file = pickle.load(f)

            self.len_file = [(length - 1) // 10 + 1 if length > 0 else 0 for length in self.len_file]

            train_len = int(len(self.len_file) * 0.99)

            if split == "train":
                self.len_file = self.len_file[:train_len]
                self.start_len = 0
            else:
                self.len_file = self.len_file[train_len:]
                self.start_len = train_len

            self.cum_len = np.cumsum(self.len_file)


            with TimingContext("get item example"):
                example = self.__getitem__(0)

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, index):
        file_index = np.argmax(self.cum_len > index)
        if file_index:
            step = index - self.cum_len[file_index - 1]
        else:
            step = index

        with TimingContext("load_image"):
            image_0_path = os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"processed_env_rand_cam_third_step_{step * 10}.jpg")
            image_1_path = os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"processed_env_cano_cam_third_step_{step * 10}.jpg")

            with open(image_0_path, 'rb') as f0, open(image_1_path, 'rb') as f1:
                jpeg_1 = f1.read()
                jpeg_0 = f0.read()
                image_0 = self.jpeg.decode(jpeg_0, pixel_format=TJPF_RGB)  #0
                image_1 = self.jpeg.decode(jpeg_1, pixel_format=TJPF_RGB)  #0

        with TimingContext("image process"):
            image_0 = image_0.astype(np.float32)
            image_1 = image_1.astype(np.float32)

            image_0 = (image_0 - 127.5) / 127.5
            image_1 = (image_1 - 127.5) / 127.5

        example = dict(
            canonical_image=image_1,
            random_image=image_0
        )

        return example


class Sim2SimDrawer(Dataset):
    def __init__(self,
                 info_file,
                 data_root,
                 split,
                 original_size=(240, 320),
                 out_size=(256, 256),
                 crop_ratio=0.95,
                 flip_p=0.5,
                 interpolation="bicubic"
                 ):
        self.data_root = data_root

        with open(info_file, "rb") as f:
            self.info_file = pickle.load(f)

        self.len_file = [(length - 1) // 10 + 1 if length > 0 else 0 for seed, suc, length in self.info_file]

        train_len = int(len(self.len_file) * 0.99)

        if split == "train":
            self.len_file = self.len_file[:train_len]
            self.start_len = 0
        else:
            self.len_file = self.len_file[train_len:]
            self.start_len = train_len

        self.cum_len = np.cumsum(self.len_file)

        self.out_size = out_size
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.transforms = [
            transforms.RandomCrop(size=[int(original_size[0] * crop_ratio), int(original_size[1] * crop_ratio)]),
            transforms.Resize(out_size, interpolation=self.interpolation),
            # transforms.RandomRotation(degrees=[-15.0, 15.0], expand=False),
            # transforms.RandomHorizontalFlip(p=flip_p),
        ]
        self.rand_only_transforms = [
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
        ]

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, index):
        file_index = np.argmax(self.cum_len > index)
        if file_index:
            step = index - self.cum_len[file_index - 1]
        else:
            step = index

        image_0 = Image.open(os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"step_{step * 10}_cam_third_rand.jpg"))
        image_1 = Image.open(os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"step_{step * 10}_cam_third_cano.jpg"))

        image_0 = np.array(image_0)
        image_1 = np.array(image_1)

        images = np.stack((image_0, image_1), axis=0)
        images = torch.from_numpy(images)  # (2, h, w, c)
        images = torch.einsum('k h w c -> k c h w', images)

        for transform in self.transforms:
            images = transform(images)

        image_0, image_1 = images[0], images[1]

        for transform in self.rand_only_transforms:
            image_0 = transform(image_0)

        image_0 = torch.einsum('c h w -> h w c', image_0)
        image_1 = torch.einsum('c h w -> h w c', image_1)

        image_0 = (image_0 / 127.5 - 1.0).float().numpy()
        image_1 = (image_1 / 127.5 - 1.0).float().numpy()

        example = dict(
            canonical_image=image_1,
            random_image=image_0
        )

        return example


class Sim2Sim_2Cam(Dataset):
    def __init__(self,
                 len_file,
                 data_root,
                 split,
                 original_size=(240, 320),
                 out_size=(256, 256),
                 crop_ratio=0.95,
                 flip_p=0.5,
                 interpolation="bicubic"
                 ):
        self.data_root = data_root

        with open(len_file, "rb") as f:
            self.len_file = pickle.load(f)

        self.len_file = [(length - 1) // 10 + 1 if length > 0 else 0 for length in self.len_file]

        train_len = int(len(self.len_file) * 0.99)

        if split == "train":
            self.len_file = self.len_file[:train_len]
            self.start_len = 0
        else:
            self.len_file = self.len_file[train_len:]
            self.start_len = train_len

        self.cum_len = np.cumsum(self.len_file)

        self.out_size = out_size
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.transforms = [
            transforms.RandomCrop(size=[int(original_size[0] * crop_ratio), int(original_size[1] * crop_ratio)]),
            transforms.Resize(out_size, interpolation=self.interpolation),
            # transforms.RandomRotation(degrees=[-15.0, 15.0], expand=False),
            # transforms.RandomHorizontalFlip(p=flip_p),
        ]
        self.rand_only_transforms = [
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
        ]

        self.random_state = np.random.RandomState(23)

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, index):
        file_index = np.argmax(self.cum_len > index)
        if file_index:
            step = index - self.cum_len[file_index - 1]
        else:
            step = index

        if self.random_state.rand() < 0.5:
            cam = "third"
        else:
            cam = "wrist"

        image_0 = Image.open(os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"env_rand_cam_{cam}_step_{step * 10}.jpg"))
        image_1 = Image.open(os.path.join(self.data_root, f"seed_{file_index + self.start_len}", f"env_cano_cam_{cam}_step_{step * 10}.jpg"))

        image_0 = np.array(image_0)
        image_1 = np.array(image_1)

        images = np.stack((image_0, image_1), axis=0)
        images = torch.from_numpy(images)  # (2, h, w, c)
        images = torch.einsum('k h w c -> k c h w', images)

        for transform in self.transforms:
            images = transform(images)

        image_0, image_1 = images[0], images[1]

        for transform in self.rand_only_transforms:
            image_0 = transform(image_0)

        image_0 = torch.einsum('c h w -> h w c', image_0)
        image_1 = torch.einsum('c h w -> h w c', image_1)

        image_0 = (image_0 / 127.5 - 1.0).float().numpy()
        image_1 = (image_1 / 127.5 - 1.0).float().numpy()

        example = dict(
            canonical_image=image_1,
            random_image=image_0
        )

        return example



class Sim2SimReal(Dataset):
    def __init__(self,
                 data_paths,
                 original_size=(480, 640),
                 out_size=(256, 256),
                 crop_ratio=60/70,
                 interpolation="bicubic"
                 ):
        self.data_paths = data_paths

        self.original_size = original_size
        self.out_size = out_size

        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.transforms = [
            transforms.CenterCrop(size=[int(original_size[0] * crop_ratio), int(original_size[1] * crop_ratio)]),
            transforms.Resize(out_size, interpolation=self.interpolation),
        ]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        path = self.data_paths[index]

        image = Image.open(path)
        image = np.array(image)

        image = torch.from_numpy(image)  # (h, w, c)
        image = torch.einsum('h w c -> c h w', image)

        for transform in self.transforms:
            image = transform(image)

        image = torch.einsum('c h w -> h w c', image)

        image = (image / 127.5 - 1.0).float().numpy()
        unknown = np.zeros_like(image)

        example = dict(
            canonical_image=unknown,
            random_image=image
        )

        return example



# class LSUNChurchesTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)
#
#
# class LSUNChurchesValidation(LSUNBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
#                          flip_p=flip_p, **kwargs)
#
#
# class LSUNBedroomsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)
#
#
# class LSUNBedroomsValidation(LSUNBase):
#     def __init__(self, flip_p=0.0, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
#                          flip_p=flip_p, **kwargs)
#
#
# class LSUNCatsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)
#
#
# class LSUNCatsValidation(LSUNBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
#                          flip_p=flip_p, **kwargs)


if __name__ == '__main__':
    data_root = "/home/zhouzhiting/Data/panda_data/sim2sim_pd_2"
    len_file = "/home/zhouzhiting/Data/panda_data/sim2sim_pd_2/cnt.pkl"
    
    data_root2 = "/home/zhouzhiting/Data/panda_data/sim2sim_pd_2_processed"
    len_file2 = "/home/zhouzhiting/Data/panda_data/sim2sim_pd_2/cnt.pkl"

    split = "train"
    # dataset = Sim2SimBase(len_file, data_root, split)
    dataset = Sim2SimBasePre(len_file2 ,data_root2, split)


    print_timing_stats()
