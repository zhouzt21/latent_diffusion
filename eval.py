import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
import imageio

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from ldm.data.sim2sim import Sim2SimReal


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    ###
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    model = instantiate_from_config(config.model)
    model = model.load_from_checkpoint(opt.resume_from_checkpoint, **config.model.params)

    # for param in model.parameters():
    #     param.requires_grad = False

    use_fp16 = True
    # dtype = torch.bfloat16
    dtype = torch.float16
    if use_fp16:
        model.model.diffusion_model.dtype = dtype
        model.to(dtype)

    # model = torch.compile(model, mode="max-autotune", fullgraph=True)
    # model._log_images = torch.compile(model._log_images)
    # model.model.diffusion_model = torch.compile(model.model.diffusion_model, mode="max-autotune", fullgraph=True)
    # print(type(model))
    # exit()

    model.cuda()
    model.eval()

    # real = True
    real = False
    save_result = True
    N = 1
    num_batch = 20
    ddim_steps = 20

    if real:
        # data_root = "/home/cuijingzhi/franka/frames/20240323_073715"
        data_root = "/home/cuijingzhi/franka/frames/20240323_074236"
        paths = ["%04d.jpg" % j for j in range(128)]
        data_paths = [os.path.join(data_root, path) for path in paths]
        # data_paths = [
        #     "/home/cuijingzhi/franka/frames/20240321_082606/0003.jpg",
        #     "/home/cuijingzhi/franka/frames/20240323_074034/0003.jpg",
        #     "/home/cuijingzhi/franka/frames/20240323_073715/0068.jpg",
        #     "/home/cuijingzhi/franka/frames/20240323_073715/0003.jpg"
        # ]
        dataset = Sim2SimReal(
            data_paths=data_paths,
            crop_ratio=60 / 70 * 0.95
        )
        eval_dataloader = DataLoader(dataset, batch_size=N, shuffle=False, drop_last=False)
    else:
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        # eval_dataloader = data.val_dataloader()
        eval_dataloader = data.train_dataloader()

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_result:
        save_root = os.path.join("tmp", "real" if real else "sim", now,
                             f"ddim_{ddim_steps}" + ("_fp16" if use_fp16 else "") + ("_b" if dtype == torch.bfloat16 else ""))
        os.makedirs(save_root, exist_ok=True)

    t_list = []

    for batch_idx, batch in enumerate(eval_dataloader):
        if batch_idx >= num_batch:
            avg_time = np.mean(t_list[1:])
            print(avg_time)
            if save_result:
                f = open(os.path.join(save_root, "time.txt"), "w")
                f.write(str(avg_time) + "\n")
            exit()
        # cano_batch = batch["canonical_image"]  # (b, 256, 256, 3)
        # rand_batch = batch["random_image"]  # (b, 256, 256, 3)
        # z, c, x, xrec = model.get_input(batch, "canonical_image", return_first_stage_outputs=True,
        #                          force_c_encode=True, bs=4)
        # print(z.shape, c.shape)
        # batch = batch[:8]
        # batch = {k: v.half() for k, v in batch.items()}
        # print([v.dtype for v in batch.values()])

        eta = 1
        start = time.time()
        # logs = model.log_images(batch, N=N, ddim_eta=eta, inpaint=False, quantize_denoised=False,
        #                         plot_denoise_rows=False, ddim_steps=20,
        #                         plot_progressive_rows=False, plot_diffusion_rows=False)
        if use_fp16:
            logs = model.log_images_(batch, N=N, ddim_eta=eta, ddim_steps=ddim_steps, dtype=dtype)
        else:
            logs = model.log_images(batch, N=N, ddim_eta=eta, inpaint=False, quantize_denoised=False,
                                    plot_denoise_rows=False, ddim_steps=ddim_steps,
                                    plot_progressive_rows=False, plot_diffusion_rows=False)
        end = time.time()
        print(end - start)
        t_list.append(end - start)

        zs = logs["z_samples"]
        s = logs["samples"]
        x = logs["inputs"]
        # xrec = logs["reconstruction"]
        xc = batch["random_image"][:N]

        # print(zs.shape, s.shape)
        # d01 = ((z - zs) ** 2).mean()
        # d00 = (((z[0] - z[1]) ** 2).mean() + ((z[0] - z[1]) ** 2).mean() + ((z[0] - z[2]) ** 2).mean()) / 3
        # d11 = (((zs[0] - zs[1]) ** 2).mean() + ((zs[0] - zs[1]) ** 2).mean() + ((zs[0] - zs[2]) ** 2).mean()) / 3
        # print(d00, d11, d01)
        # print(logs.keys())
        # print(x.shape, xrec.shape, s.shape, zs.shape, xc.shape)


        def tensor2img(tensor, transpose=True):
            img = torch.clamp(tensor, -1., 1.).detach().float().cpu()
            img = ((img + 1) * 127.5).numpy()
            if transpose:
                img = img.transpose(1, 2, 0)  # (c, h, w) -> (h, w, c)
            img = np.clip(0, 255, np.round(img)).astype(np.uint8)
            return img


        for i in range(N):
            xi = tensor2img(x[i])
            # xreci = tensor2img(xrec[i])
            si = tensor2img(s[i])
            xci = tensor2img(xc[i], transpose=False)
            gap = np.ones((256, 2, 3), dtype=np.uint8) * 255
            if real:
                cat_i = np.concatenate([xci, gap, si], axis=1)
            else:
                cat_i = np.concatenate([xci, gap, si, gap, xi], axis=1)
                if save_result:
                    imageio.imsave(os.path.join(save_root, f"x_{batch_idx}_{i}.jpg"), xi)
                    # imageio.imsave(os.path.join(save_root, f"xrec_{batch_idx}_{i}.jpg"), xreci)

            if save_result:
                imageio.imsave(os.path.join(save_root, f"s_{batch_idx}_{i}.jpg"), si)
                imageio.imsave(os.path.join(save_root, f"xc_{batch_idx}_{i}.jpg"), xci)
                imageio.imsave(os.path.join(save_root, f"cat_{batch_idx}_{i}.jpg"), cat_i)
