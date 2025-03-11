import json
import os

import PIL

import imageio
from flask import Flask, jsonify, request
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config

app = Flask(__name__)

stamp = "2025-03-04T22-20-00"
logdir = os.path.join("/home/zhouzhiting/Data/panda_data/latent_diffusion_logs", f"{stamp}_sim2sim-kl-8")
ckpt_path = os.path.join(logdir, "checkpoints", "epoch=000013.ckpt")
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

# original_size = (480, 640) # real
original_size = (240, 320)  # sim

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


def process_data(image_list):
    images = np.stack(image_list, axis=0)  # (B, h, w, c)
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
    img = torch.clamp(tensor, -1., 1.).detach().float().cpu()
    img = ((img + 1) * 127.5).numpy()
    if transpose:
        img = img.transpose(0, 2, 3, 1)  # (k, c, h, w) -> (k, h, w, c)
    img = np.clip(0, 255, np.round(img)).astype(np.uint8)
    return img


@app.route("/ldm", methods=["GET", "POST"])
def handle_request():
    """"
        process the image only.
    """
    if request.method == "GET":
        response = {"message": "GET response"}
        return jsonify(response)
    elif request.method == "POST":
        try:
            os.makedirs("tmp", exist_ok=True)
            files = request.files

            image_list = []
            for file in files.values():
                # image = np.fromfile(file, np.uint8)
                # image = imageio.imread(file)
                file.save(os.path.join("tmp", "origin.jpg"))
                image = imageio.imread(os.path.join("tmp", "origin.jpg"))
                # image = Image.open(file)
                # image = np.array(image)
                image_list.append(image)

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

            imageio.imsave(os.path.join("tmp", f"sample.jpg"), samples[0])

            response = dict(
                samples=samples.tolist()
            )

            return jsonify(response)

        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 400


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
