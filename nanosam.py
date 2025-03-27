# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tqdm
import torch
import argparse
# from nanosam.utils.predictor import load_image_encoder_engine
# from nanosam.models import create_model, list_models
# from nanosam.datasets.image_folder import ImageFolder
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torch.distributed as dist
import builtins
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
# from mmpretrain.models import VisionTransformer
# from mmpretrain.models import DistilledVisionTransformer
# from mmpretrain.models import TransformerEncoderLayer

from torch.profiler.profiler import tensorboard_trace_handler
import torch.nn as nn
from functools import partial
import models_pretrain

from util.load_dataset import get_image_webdataset, get_image_webdataloader, get_image_iterator
import torchvision.transforms as transforms
import PIL
import math

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (dist.get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    
def count_parameters(model):
    total_params = 0
    total_grad_params = 0
    total_non_grad_params = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            total_grad_params += param_count
        else:
            total_non_grad_params += param_count
    
    print(f"Total parameters: {total_params}, Trainable parameters: {total_grad_params}, Non-trainable parameters: {total_non_grad_params}")

class ResizeAndPadToAspectRatio:
    def __init__(self, target_aspect_ratio=0.75, target_size=(1024, 768)):
        self.target_aspect_ratio = target_aspect_ratio
        self.target_size = target_size
        self.resize_transform = transforms.Resize(target_size)
    
    def __call__(self, img): # img: PIL.Image
        width, height = img.size

        current_aspect_ratio = width / height

        if current_aspect_ratio > self.target_aspect_ratio: # the width is enough, Pad height
            new_width = width
            new_height = int(new_width / self.target_aspect_ratio)
            padding_top = (new_height - height) // 2
            padding_bottom = new_height - height - padding_top
            padding = (0, padding_top, 0, padding_bottom)
        else:
            new_height = height
            new_width = int(new_height * self.target_aspect_ratio)
            padding_left = (new_width - width) // 2
            padding_right = new_width - width - padding_left
            padding = (padding_left, 0, padding_right, 0)

        padded_img = transforms.Pad(padding)(img)

        resized_img = self.resize_transform(padded_img)
        
        return resized_img
    
class DataAugmentationv4(object):
    # Two base images (1024x768) for teacher model and student model. Non-geometrical data argument.
    # N croped images (128x96) for student and ema student.
    def __init__(self, size, crop_size, global_crops_scale, local_crops_scale, local_crops_number):
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),  # 3 is bicubic
            ResizeAndPadToAspectRatio(target_aspect_ratio=0.75, target_size=size),
            # transforms.RandomHorizontalFlip(),
            # flip_and_color_jitter,
            # color_jitter,
            # misc.GaussianBlur(1.0),
            # misc.Solarization(0.2),
            normalize]
        )

    def __call__(self, image):
        image = PIL.Image.fromarray(image)
        aug_img_1 = self.global_transform(image)
        return [aug_img_1]

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="The path to images to use for distillation")
    parser.add_argument("--output_dir", type=str, help="The directory to store checkpoints and training visualizations.")
    parser.add_argument("--model_name", type=str, default="resnet18", help="The NanoSAM model name.")
    parser.add_argument("--student_size", type=int, default=1024, help="The size of image to feed to the student during distillation.")
    parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images per epoch.  Helpful for quick training runs when experimenting.")
    parser.add_argument("--num_epochs", type=int, default=200, help="The number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of data loader workers.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help='The learning rate.')
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1", "mse"],
        help="The loss function to use for distillation.")
    parser.add_argument("--teacher_image_encoder_engine", 
        type=str, 
        default="data/mobile_sam_image_encoder_bs16.engine",
        help="The path to the image encoder engine to use as a teacher model."
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    
    ### DDP init
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    device = torch.device(args.rank)
    
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    ### END
    
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 12345)) # 指定主机和端口
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client() # 等待 VSCode 连接
        debugpy.breakpoint() # 设置断点
        print("Debugger is attached.")    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_encoder_cnn = models_pretrain.__dict__["saip_kd_vit_tiny_patch16_sapiens_0_3b_conv"](teacher_patch_resolution=(64,48))
    image_encoder_cnn = image_encoder_cnn.cuda()
    count_parameters(image_encoder_cnn)
    
    loss_function = F.huber_loss

    optimizer = torch.optim.Adam(image_encoder_cnn.parameters(), lr=args.learning_rate)

    transform_train = DataAugmentationv4((512,384),
        (256,128),
        1,
        1,
        2)
    dataset_train, dataset_train_len = get_image_webdataset(
        dataset_root="/mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_theia_data",
        feature_models=["sapiens_0.3b_coco_best_coco_AP_796"],
        dataset = 'HD1M',
        split="train",
        image_transform=transform_train,
        feature_norm=False, # todo
        seed=args.seed,
        shuffle=True,
        world_size=dist.get_world_size()
    )

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        image_encoder_cnn.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    image_encoder_cnn = torch.nn.parallel.DistributedDataParallel(image_encoder_cnn, device_ids=[args.gpu], find_unused_parameters=True)

    for epoch in range(start_epoch, args.num_epochs):

        epoch_loss = 0.

        data_loader_train = get_image_webdataloader(
            dataset_train,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            shuffle=True,
            shuffle_buffer_size=1024,
            seed=args.seed + dist.get_rank() * 100 + epoch
        )
        data_loader_iter = get_image_iterator(data_loader_train)
        len_loader = math.ceil(dataset_train_len / args.batch_size / dist.get_world_size())
        pbar = tqdm.tqdm(range(len_loader), desc=f'Epoch {epoch}', disable=dist.get_rank()!=0)
        log_every = 20
        for idx in pbar:
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = get_image_iterator(data_loader_iter)
                batch = next(data_loader_iter)
            image = batch['image'][0].cuda()
            image_cnn = image
            with torch.cuda.amp.autocast():
                features = batch["sapiens_0.3b_coco_best_coco_AP_796"]['embedding'].cuda(non_blocking=True)

                optimizer.zero_grad()
                student_out = image_encoder_cnn(image_cnn, meta={"region_imgs":None}, with_cls_token=False, extract_regions=False) # [16,192,64,48]
                output = student_out["aligned_patch_feats"]

            loss = loss_function(output, features)

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            if idx%log_every == 0:
                print(loss.item())
            # profiler.step()

        epoch_loss /= len_loader
        print(f"{epoch} - {epoch_loss}")
        
        pbar.set_postfix({'avg_loss': epoch_loss})
        pbar.close()

        if dist.get_rank() == 0:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(f"{epoch} - {epoch_loss}\n")

            torch.save({
                "model": image_encoder_cnn.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, checkpoint_path)