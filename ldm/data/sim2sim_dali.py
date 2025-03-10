from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import os

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



import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.files = []

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def feed(self, inputs):
        self.files.extend(inputs)

    def __next__(self):
        batch = []
        if len(self.files) < self.batch_size:
            raise StopIteration()
   
        for _ in range(self.batch_size):
            jpeg_filename = self.files.pop()
            f = open(jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        return batch

class SimDALIPipeline(Pipeline):
    def __init__(self, batch_size=2, num_threads=4, device_id=0,
                 original_size=(240, 320), out_size=(256, 256), crop_ratio=0.95):
        super().__init__(batch_size, num_threads, device_id, 
                        exec_pipelined=False, exec_async=False)
        
        self.original_size = original_size
        self.out_size = out_size
        self.crop_size = [int(original_size[0] * crop_ratio), 
                         int(original_size[1] * crop_ratio)]
        
        self.eii = ExternalInputIterator(batch_size)

    def define_graph(self):
        # 读取 JPEG
        jpegs = fn.external_source(source=self.eii, device="cpu")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        
        # Random Crop
        images = fn.random_resized_crop(
            images,
            size=self.crop_size,
            random_area=(0.9025, 1.0),  # 0.95 * 0.95
            random_aspect_ratio=(1.0, 1.0),
            device="gpu"
        )
        
        # Resize
        images = fn.resize(
            images,
            size=self.out_size,
            device="gpu"
        )
        
        # 颜色增强
        images = fn.color_twist(
            images,
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            device="gpu"
        )
        
        # 转换为浮点数并标准化
        images = fn.cast(images, dtype=types.FLOAT)
        images = fn.normalize(
            images,
            mean=127.5,
            stddev=127.5,
            device="gpu"
        )
        
        # CHW 格式
        images = fn.transpose(images, perm=[2, 0, 1])
        
        return images


class Sim2SimBaseDALI(Dataset):
    def __init__(self, len_file, data_root, split, batch_size=2, num_threads=4, device_id=0, **kwargs):
        super().__init__()
        
        with TimingContext("dataset_initialization"):
            self.data_root = data_root
            
            # 初始化索引
            with open(len_file, "rb") as f:
                self.len_file = pickle.load(f)
                
            self.len_file = [(length - 1) // 10 + 1 if length > 0 else 0 
                            for length in self.len_file]
            
            train_len = int(len(self.len_file) * 0.99)
            
            if split == "train":
                self.len_file = self.len_file[:train_len]
                self.start_len = 0
            else:
                self.len_file = self.len_file[train_len:]
                self.start_len = train_len
                
            self.cum_len = np.cumsum(self.len_file)
            
            # 初始化DALI pipeline
            with TimingContext("dali_pipeline_initialization"):
                self.pipeline = SimDALIPipeline(
                    batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=device_id,
                    **kwargs
                )
            with TimingContext("dali_pipeline_build"):
                self.pipeline.build()
            
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
            
        # 准备图像路径
        image_paths = [
            os.path.join(self.data_root, 
                        f"seed_{file_index + self.start_len}", 
                        f"env_rand_cam_third_step_{step * 10}.jpg"),
            os.path.join(self.data_root,
                        f"seed_{file_index + self.start_len}",
                        f"env_cano_cam_third_step_{step * 10}.jpg")
        ]
        
        # 通过DALI处理图像
        with TimingContext("dali_pipeline_run"):
            self.pipeline.eii.feed(image_paths)
            pipe_out = self.pipeline.run()
            images = pipe_out[0].as_cpu().as_array()
        
        return {
            "canonical_image": images[1],
            "random_image": images[0]
        }
