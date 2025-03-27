# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Organize imagefolder-like images (ImageNet) to webdataset format."""

import argparse
import glob
import os
import shutil
import tarfile
from io import BytesIO

import numpy as np
import webdataset as wds
from numpy.typing import NDArray
from PIL import Image
# from torchvision.transforms.v2 import Compose, Resize
from torchvision.transforms import Compose, Resize, RandomResizedCrop
from typing import Dict, List, Tuple
import torch
import torchvision.transforms as transforms

from transfroms import ResizeAndPadToAspectRatio

def denormalize(tensor, mean=None, std=None):
    """
    对经过 Normalize 的 tensor 进行反归一化。
    参数:
        tensor: 输入的张量，形状为 [C, H, W] 或 [B, C, H, W]
        mean: 归一化时使用的均值 (tuple)
        std: 归一化时使用的标准差 (tuple)
    返回:
        反归一化后的张量
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    mean = torch.tensor(mean).view(-1, 1, 1)  # 转换为形状 [C, 1, 1]
    std = torch.tensor(std).view(-1, 1, 1)    # 转换为形状 [C, 1, 1]
    return tensor * std + mean  # 反归一化公式

def check_existing_shard(path: str) -> bool:
    """Check the integrity of the existing webdataset shard.

    Args:
        path (str): path to the webdataset shard.

    Returns:
        bool: True for complete shard.
            False for non-existing or broken shard.
    """
    # try:
    #     tarf = tarfile.open(path)
    #     for _ in tarf.getmembers():
    #         pass
    # except (ValueError, tarfile.ReadError, tarfile.CompressionError) as e:
    #     print(e)
    #     return False
    # return True
    return os.path.exists(path)


def create_shard(
    args: argparse.Namespace,
    shard_idx: int,
    shard_path: str,
    remote_shard_path: str,
    frames: List[Tuple[NDArray, str]],
) -> None:
    """Create a webdataset shard.

    Args:
        args (argparse.Namespace): arguments.
        shard_idx (int): index of this shard.
        shard_path (str): (local) path to save the shard.
        remote_shard_path (str): final destination (remote) to save the shard.
        frames (list[tuple[NDArray, str]]): images to save in this shard.
    """
    if check_existing_shard(remote_shard_path):
        print(f"creating {args.dataset} shard {shard_idx:06d} - check pass, skip\r", end="")
        return
    print(f"creating {args.dataset} shard {shard_idx:06d}\r", end="")
    if shard_path is None:
        shard_path = remote_shard_path
    with wds.TarWriter(shard_path) as tar_writer:
        for i, (image, basename) in enumerate(frames):
            image_out = BytesIO()
            np.save(image_out, image)
            sample = {"__key__": basename, "image": image_out.getvalue()}
            tar_writer.write(sample)
            if (i + 1) % 20 == 0:
                print(f"creating {args.dataset} shard {shard_idx:06d} - {(i+1) * 100 // len(frames):02d}%\r", end="\r")
    if shard_path != remote_shard_path:
        shutil.move(shard_path, remote_shard_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--imagenet-raw-path", type=str)
    parser.add_argument("--tmp-shard-path", type=str, default="None")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    args = parser.parse_args()
    # match args.dataset:
    #     case "imagenet":
    #         IMAGE_DATASET_RAW_DIR = args.imagenet_raw_path
    #     case _:
    #         raise NotImplementedError(f"{args.dataset} is not supported")
    IMAGE_DATASET_RAW_DIR = args.imagenet_raw_path

    if args.tmp_shard_path == "None":
        TMP_SHARD_PATH = None
    else:
        TMP_SHARD_PATH = os.path.join(args.tmp_shard_path, args.dataset)
        if not os.path.exists(TMP_SHARD_PATH):
            os.makedirs(TMP_SHARD_PATH)

    OUTPUT_SHARD_PATH = os.path.join(args.output_path, args.dataset, 'images')
    if not os.path.exists(OUTPUT_SHARD_PATH):
        os.makedirs(OUTPUT_SHARD_PATH, exist_ok=True)

    if args.split == "train":
        image_paths = sorted(glob.glob(f"{IMAGE_DATASET_RAW_DIR}/{args.split}/*/*.jpg"))
    else:
        image_paths = sorted(glob.glob(f"{IMAGE_DATASET_RAW_DIR}/{args.split}/*/*.jpg"))
    print(len(image_paths))
    # transform = Compose([Resize((224, 224), antialias=True)])
    # transform = Compose([RandomResizedCrop(size=(1024,768), scale=(1,1), interpolation=3, ratio=(0.75,0.75), antialias=True)])
    # transform = None
    transform = Compose([
        ResizeAndPadToAspectRatio(target_aspect_ratio=0.75, target_size=(1024,768)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if transform is None:
        print('Transform is None!')
    else:
        print(transform)

    shard_idx = 0
    shard_buffer: list[tuple[NDArray, str]] = []
    for image_path in image_paths:
        # basename = image_path.split("/")[-1].split(".")[0]
        basename = os.path.splitext(image_path)[0]
        ori_image = Image.open(image_path)
        if transform:
            image = np.array(transform(ori_image))
        else:
            image = np.array(ori_image)
        # 可视化
        # scale = transform.transforms[0].scale
        # ratio = transform.transforms[0].ratio
        # size = transform.transforms[0].size
        # save_dir  = f'temp/feature_extraction/{size[0]}x{size[1]}_{scale[0]}-{scale[1]}_{ratio[0]}-{ratio[1]}'
        # save_dir  = f'temp/feature_extraction/no_transform'
        save_dir = f'temp/feature_extraction/my_transform'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from torchvision.transforms import ToPILImage
        import matplotlib.pyplot as plt
        save_path = f'{save_dir}/{basename.split("/")[-1]}.jpg'
        fig, axes = plt.subplots(1, 2, figsize=(1 * 3, 2 * 3))
        axes = axes.flatten()

        w, h = ori_image.size
        axes[0].imshow(ori_image)
        axes[0].axis('off')
        axes[0].set_title(f"ori_{h}x{w}", fontsize=10)

        h, w, _ = image.shape
        plt_image = ToPILImage()(image)
        axes[1].imshow(plt_image)
        axes[1].axis('off')
        axes[1].set_title(f"after_transfrom_{h}x{w}", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(image.shape)
        # end
        shard_buffer.append((image, basename))
        if len(shard_buffer) % 20 == 0:
            print(f"shard {shard_idx: 04d} frames {len(shard_buffer)}\r", end="\r")
        if len(shard_buffer) == args.samples_per_shard:
            shard_fn = f"{args.dataset}_{args.split}-{shard_idx:06d}-{args.split}.tar"
            local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None # 本地路径
            remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn) # 远程路径
            create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)
            shard_buffer = []
            shard_idx += 1

    shard_fn = f"{args.dataset}_{args.split}-{shard_idx:06d}-{args.split}.tar"
    local_shard_path = os.path.join(TMP_SHARD_PATH, shard_fn) if TMP_SHARD_PATH else None
    remote_shard_path = os.path.join(OUTPUT_SHARD_PATH, shard_fn)
    if len(shard_buffer) > 0:
        create_shard(args, shard_idx, local_shard_path, remote_shard_path, shard_buffer)


if __name__ == "__main__":
    main()
