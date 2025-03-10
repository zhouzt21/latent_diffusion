import os
import numpy as np
import PIL
from PIL import Image
import torch
from torchvision import transforms
from turbojpeg import TurboJPEG, TJPF_RGB
from tqdm import tqdm
import pickle
from collections import defaultdict
import threading
import time

# 复用计时统计代码
timing_stats = defaultdict(list)
timing_lock = threading.Lock()

class TimingContext:
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

def get_seed_number(seed_dir):
    """从种子目录名中提取数字"""
    try:
        return int(seed_dir.split('_')[1])
    except (IndexError, ValueError):
        return float('inf')  # 对于无效的目录名返回无穷大


def preprocess_images(data_root, out_root, original_size=(240, 320), 
                     out_size=(256, 256), crop_ratio=0.95):
    """预处理数据集中的所有图片"""
    
    # 创建 TurboJPEG 实例
    jpeg = TurboJPEG()
    
    # 创建变换
    interpolation = PIL.Image.BICUBIC
    transforms_list = [
        transforms.RandomCrop(size=[int(original_size[0] * crop_ratio), 
                                  int(original_size[1] * crop_ratio)]),
        transforms.Resize(out_size, interpolation=interpolation)
    ]
    
    # 创建颜色增强
    color_transforms = [
        transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
    ]
    

    # 获取并排序所有种子文件夹
    seed_dirs = [d for d in os.listdir(data_root) if d.startswith('seed_')]
    seed_dirs.sort(key=get_seed_number)  # 按数字大小排序
    
    print(f"Found {len(seed_dirs)} seed directories.")

    # 遍历所有种子文件夹
    for seed_dir in tqdm(seed_dirs):
        if not seed_dir.startswith('seed_'):
            continue
            
        seed_path = os.path.join(data_root, seed_dir)
        out_seed_path = os.path.join(out_root, seed_dir)

        if os.path.exists(out_seed_path):
            print(f"{out_seed_path} already exists, skipping.")
            continue

        os.makedirs(out_seed_path, exist_ok=True)
        
        # 处理每个时间步的图片
        for img_file in os.listdir(seed_path):
            if not img_file.endswith('.jpg'):
                continue
                
            with TimingContext("load_and_process"):
                # 读取图片
                img_path = os.path.join(seed_path, img_file)
                with open(img_path, 'rb') as f:
                    jpeg_data = f.read()
                    image = jpeg.decode(jpeg_data, pixel_format=TJPF_RGB)
                
                # 转换为张量
                image = np.array(image)
                image = torch.from_numpy(image)  # (h, w, c)
                image = torch.einsum('h w c -> c h w', image)
                
                # 应用变换
                for transform in transforms_list:
                    image = transform(image)
                    
                # 如果是随机视角图像，应用颜色增强
                if 'rand' in img_file:
                    for transform in color_transforms:
                        image = transform(image)
                
                # 转换回HWC格式
                image = torch.einsum('c h w -> h w c', image)
                
                # 直接使用uint8格式保存
                image = image.numpy().astype(np.uint8)
                
                # 保存处理后的图片为JPEG
                out_path = os.path.join(out_seed_path, f"processed_{img_file}")

                with open(out_path, 'wb') as f:
                    f.write(jpeg.encode(image, quality=95, pixel_format=TJPF_RGB))


from tqdm import tqdm
import threading
from queue import Queue
from collections import defaultdict
import time

def process_images_mt(root_path, out_root, num_threads=64, target_folders=None):
    """多线程版本的图像预处理函数"""
    
    def worker(queue, worker_id):
        # 每个线程创建自己的处理工具
        jpeg = TurboJPEG()
        interpolation = PIL.Image.BICUBIC
        transforms_list = [
            transforms.RandomCrop(size=[int(240 * 0.95), int(320 * 0.95)]),
            transforms.Resize((256, 256), interpolation=interpolation)
        ]
        color_transforms = [
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
        ]
        
        while True:
            item = queue.get()
            if item is None:
                break
                
            seed_dir, img_file = item
            try:
                seed_path = os.path.join(root_path, seed_dir)
                out_seed_path = os.path.join(out_root, seed_dir)
                out_path = os.path.join(out_seed_path, f"processed_{img_file}")

                # 检查输出文件是否已存在
                if os.path.exists(out_path):
                    queue.task_done()
                    continue

                with TimingContext(f"worker_{worker_id}_process"):
                    # 读取图片
                    img_path = os.path.join(seed_path, img_file)
                    with open(img_path, 'rb') as f:
                        jpeg_data = f.read()
                        image = jpeg.decode(jpeg_data, pixel_format=TJPF_RGB)
                    
                    # 转换为张量并处理
                    image = torch.from_numpy(image).permute(2, 0, 1)
                    
                    # 应用变换
                    for transform in transforms_list:
                        image = transform(image)
                    
                    # 对随机视角图像应用颜色增强
                    if 'rand' in img_file:
                        for transform in color_transforms:
                            image = transform(image)
                    
                    # 转换回HWC格式
                    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
                    
                    # 保存处理后的图片
                    os.makedirs(out_seed_path, exist_ok=True)
                    
                    with open(out_path, 'wb') as f:
                        f.write(jpeg.encode(image, quality=95, pixel_format=TJPF_RGB))
                    
            except Exception as e:
                print(f"线程-{worker_id} 处理出错 {img_path}: {str(e)}")
            finally:
                queue.task_done()
    
    # 获取所有数据目录
    seed_dirs = [d for d in os.listdir(root_path) if d.startswith('seed_')]

    # 过滤掉 seed number 小于 1488 的目录
    # seed_dirs = [d for d in seed_dirs if get_seed_number(d) >= 1488]
    seed_dirs.sort(key=get_seed_number)

    if target_folders is not None:
        target_folders = set(target_folders)
        seed_dirs = [d for d in seed_dirs if os.path.join(root_path, d) in target_folders]
    
    print(f"找到 {len(seed_dirs)} 个种子目录")
    
    # 创建任务队列
    task_queue = Queue()
    
    # 收集所有任务
    total_tasks = 0
    for seed_dir in seed_dirs:
        seed_path = os.path.join(root_path, seed_dir)
        out_seed_path = os.path.join(out_root, seed_dir)
        
        # if os.path.exists(out_seed_path):
        #     print(f"跳过已存在的目录: {out_seed_path}")
        #     continue
            
        # for img_file in os.listdir(seed_path):
        #     if not img_file.endswith('.jpg'):
        #         continue
        #     task_queue.put((seed_dir, img_file))
        #     total_tasks += 1
    
        for img_file in os.listdir(seed_path):
            if not img_file.endswith('.jpg'):
                continue
                
            out_path = os.path.join(out_seed_path, f"processed_{img_file}")
            if os.path.exists(out_path):
                continue
                
            task_queue.put((seed_dir, img_file))
            total_tasks += 1

    print(f"总共有 {total_tasks} 个图像需要处理")
    
    # 创建工作线程
    threads = []
    for i in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(task_queue, i),
            name=f"Worker-{i}"
        )
        t.start()
        threads.append(t)
    
    # 添加结束标记
    for _ in range(num_threads):
        task_queue.put(None)
    
    # 等待所有任务完成
    with tqdm(total=total_tasks, desc="处理进度") as pbar:
        last_tasks = task_queue.qsize()
        while not task_queue.empty():
            current_tasks = task_queue.qsize()
            done_tasks = last_tasks - current_tasks
            if done_tasks > 0:
                pbar.update(done_tasks)
                last_tasks = current_tasks
            time.sleep(0.1)
    
    # 等待所有线程结束
    task_queue.join()
    for t in threads:
        t.join()
    
    print("所有图像处理完成!")

def main():
    data_root = "/home/zhouzhiting/Data/panda_data/sim2sim_pd_2"
    out_root = "/home/zhouzhiting/Data/panda_data/sim2sim_pd_2_processed"
    
    # 创建输出目录
    os.makedirs(out_root, exist_ok=True)
    
    # 预处理图片
    # preprocess_images(data_root, out_root)
    

    # 预处理图片，设置线程数
    num_workers = 1
    print(f"Using {num_workers} workers")
    process_images_mt(data_root, out_root, num_threads=num_workers)
    

    # 打印处理时间统计
    print("\n===== 预处理时间统计 =====")
    for operation, times in sorted(timing_stats.items()):
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"{operation:<30} - 平均: {avg_time*1000:.2f}ms, "
                  f"最大: {max_time*1000:.2f}ms, "
                  f"最小: {min_time*1000:.2f}ms, "
                  f"调用次数: {len(times)}")

if __name__ == "__main__":
    main()