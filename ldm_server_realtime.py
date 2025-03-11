import json
import os
import time

import PIL
from PIL import Image
import imageio
from flask import Flask, jsonify, request
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config

import requests

app = Flask(__name__)

############## init model ##############
# stamp = "2024-04-03T12-03-36"
stamp = "2025-03-04T22-20-00"
logdir = os.path.join("/home/zhouzhiting/Data/panda_data/latent_diffusion_logs", f"{stamp}_sim2sim-kl-8")
# ckpt_path = os.path.join(logdir, "checkpoints", "epoch=000007.ckpt")
ckpt_path = os.path.join(logdir, "checkpoints", "epoch=000007-v1.ckpt")
cfg_path = os.path.join(logdir, "configs", f"{stamp}-project.yaml")
config = OmegaConf.load(cfg_path)

model = instantiate_from_config(config.model)
model = model.load_from_checkpoint(ckpt_path, **config.model.params)
seed_everything(23)

use_fp16 = True
# dtype = torch.bfloat16
dtype = torch.float16
if use_fp16:
    model.model.diffusion_model.dtype = dtype
    model.to(dtype)

model.cuda()
model.eval()

original_size = (480, 640)
out_size = (256, 256)
crop_ratio = 0.95
interpolation = "bicubic"

interpolation = {
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
}[interpolation]

transform = [
    transforms.CenterCrop(size=[int(original_size[0] * crop_ratio), int(original_size[1] * crop_ratio)]),
    transforms.Resize(out_size, interpolation=interpolation),
]

eta = 1
ddim_steps = 20

cameras = ['third']

################################


def process_data(image_list):
    """
        process the data for ldm model.
        input: image_list: [M*(h, w, c)],  uint8, np,
                for real image, original_size = (480, 640)
        output: dict{
                canonical_image: (B, c, h, w),  normalized in [-1,1], float, torch
                random_image: (B, c, h, w),  normalized in [-1,1], float, torch
            }
    """
    images = np.stack(image_list, axis=0)  # (M, h, w, c)
    images = torch.from_numpy(images)
    images = torch.einsum('k h w c -> k c h w', images)

    for t in transform:
        images = t(images)

    images = torch.einsum('k c h w -> k h w c', images)

    images = (images / 127.5 - 1.0).float()
    unknown = torch.zeros_like(images)

    example = dict(
        canonical_image=unknown.cuda(),
        random_image=images.cuda()
    )
    return example


def tensor2img(tensor, transpose=True):
    """
        convert tensor to numpy image.
        input: tensor: (M, c, h, w),  normalized in [-1,1], float, torch
        output: img: (M, h, w, c),  uint8, np
    """
    img = torch.clamp(tensor, -1., 1.).detach().float().cpu()
    img = ((img + 1) * 127.5).numpy()
    if transpose:
        img = img.transpose(0, 2, 3, 1)  # (k, c, h, w) -> (k, h, w, c)
    img = np.clip(0, 255, np.round(img)).astype(np.uint8)
    return img


@app.route("/ldm_real", methods=["GET", "POST"])
def handle_request():
    if request.method == "GET":
        response = {"message": "GET response"}
        return jsonify(response)
    elif request.method == "POST":
        try:
            os.makedirs("tmp", exist_ok=True)

            # Send request to robot server
            robot_server_url = "http://??/robot"
            robot_response = requests.post(robot_server_url)
            robot_response.raise_for_status()   
            robot_data = robot_response.json()

            tcp_pose = robot_data["tcp_pose"]
            gripper = robot_data["gripper_width"]
            print("tcp_pose: ", tcp_pose.shape)
            print("gripper: ", gripper.shape)

            image_list = []
            # just one timesteps
            for cam in cameras: 
                image_list.append(robot_data[f"{cam}-rgb"])

            batch = process_data(image_list)
            N = len(image_list)

            if use_fp16:
                logs = model.log_images_(batch, N=N, ddim_eta=eta, ddim_steps=ddim_steps, dtype=dtype)
            else:
                logs = model.log_images(batch, N=N, ddim_eta=eta, inpaint=False, quantize_denoised=False,
                                        plot_denoise_rows=False, ddim_steps=ddim_steps,
                                        plot_progressive_rows=False, plot_diffusion_rows=False)

            samples = logs["samples"]
            samples = tensor2img(samples)

            response = dict(
                samples=samples.tolist(), # (M, h, w, c), uint8, np
                tcp_pose=tcp_pose,
                gripper_width=gripper,
            )

            return jsonify(response)

        except requests.exceptions.HTTPError as http_err:
            error_msg = f"LDM服务器HTTP错误: {http_err}"
            return jsonify({"error": error_msg}), http_err.response.status_code
        except requests.exceptions.ConnectionError as conn_err:
            error_msg = f"无法连接到LDM服务器: {conn_err}"
            return jsonify({"error": error_msg}), 503
        except requests.exceptions.Timeout as timeout_err:
            error_msg = f"LDM服务器请求超时: {timeout_err}"
            return jsonify({"error": error_msg}), 504
        except requests.exceptions.RequestException as err:
            error_msg = f"请求LDM服务器时发生错误: {err}"
            return jsonify({"error": error_msg}), 500
        except Exception as e:
            error_msg = f"处理请求时发生未知错误: {str(e)}"
            return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    # app.run(host="localhost", port="9966")
    app.run(host="0.0.0.0", port="9977")

    # import requests
    # paths = ["obs.jpg"]
    # files = {str(i): open(path, 'rb') for i, path in enumerate(paths)}
    # response = requests.post("http://localhost:9966/ldm", files=files)
    # response = response.json()
    # samples = response["samples"]
    # samples = np.array(samples, dtype=np.uint8)
