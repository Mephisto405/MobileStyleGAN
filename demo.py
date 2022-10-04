import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import utils

from core.distiller import Distiller
from core.model_zoo import model_zoo
from core.utils import load_cfg, load_weights, tensor_to_img


def main(args):
    cfg = load_cfg(args.cfg)
    distiller = Distiller(cfg)
    if args.ckpt != "":
        ckpt = model_zoo(args.ckpt)
        load_weights(distiller, ckpt["state_dict"])

    var = torch.randn(args.n_sample, distiller.mapping_net.style_dim)
    img_s = distiller(var, truncated=args.truncated, generator=args.generator)
    utils.save_image(
        img_s,
        f"{args.generator}.png",
        nrow=int(args.n_sample ** 0.5),
        normalize=True,
        range=(-1, 1),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/mobile_stylegan_ffhq.json",
        help="path to config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="mobilestylegan_ffhq.ckpt",
        help="path to checkpoint",
    )
    parser.add_argument("--truncated", action="store_true", help="use truncation mode")
    parser.add_argument(
        "--generator",
        type=str,
        default="student",
        help="generator mode: [student|teacher]",
    )
    parser.add_argument("--n_sample", type=int, default=25, help="number of samples")
    args = parser.parse_args()
    main(args)
