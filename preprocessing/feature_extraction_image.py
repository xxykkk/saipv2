import sys
sys.path.append('/mnt/hdd3/wangxuanhan/research/vfm_research/learning_research/saipv2')
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from preprocessing.utils import get_feature_outputs, get_model, get_models

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Vision Transformer features")
    parser.add_argument("--model_path", type=str, 
                        default='images', 
                        help="Path to the ViT model weights")
    parser.add_argument("--image_folder", type=str, 
                        default='data/HD1M', 
                        help="Path to the folder containing images")
    parser.add_argument("--output_folder", type=str, 
                        default='/mnt/hdd4/wangxuanhan/datasets/preprocessed_HD1M', 
                        help="Path to the output folder to save features")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing images")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser.parse_args()

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

class ImageDataset(Dataset):
    def __init__(self, image_folder, split='train', transform=None):
        self.image_folder = os.path.join(image_folder, split)
        if not os.path.exists(self.image_folder):
            raise ValueError(f"{self.image_folder} not exites!")
        assert transform is not None
        self.transform = transform
        # self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
        # self.r_sub_folder = os.listdir(self.image_folder)
        # self.image_paths = []
        # for folder in self.r_sub_folder:
        #     a_sub_folder = os.path.join(self.image_folder, folder)
        #     for f in os.listdir(a_sub_folder):
        #         if f.endswith(('jpg', 'jpeg', 'png')):
        #             self.image_paths.append(os.path.join(a_sub_folder, f))
        self.image_paths = []
        for root, dirs, files in os.walk(self.image_folder, followlinks=True):
            for f in files:
                if f.endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(root, f))
        
        if len(self.image_paths) == 0: # 1218136 - 41056 = 1177080
            raise ValueError(f"No image files found in {self.image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_rgb = Image.open(image_path).convert("RGB")
        images = self.transform(image_rgb)
        # # 可视化
        # # scale = self.transform.transforms[0].scale
        # # ratio = self.transform.transforms[0].ratio
        # # size = self.transform.transforms[0].size
        # # save_dir  = f'temp/my_feature_extraction/{size[0]}x{size[1]}_{scale[0]}-{scale[1]}_{ratio[0]}-{ratio[1]}'
        # save_dir = 'temp'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # from torchvision.transforms import ToPILImage
        # import matplotlib.pyplot as plt
        # save_path = f'{save_dir}/output_image.jpg'
        # fig, axes = plt.subplots(1, 2, figsize=(1 * 3, 2 * 3))
        # axes = axes.flatten()

        # w, h = image_rgb.size
        # axes[0].imshow(image_rgb)
        # axes[0].axis('off')
        # axes[0].set_title(f"ori_{h}x{w}", fontsize=10)

        # if len(images.shape) == 3:
        #     c, h, w = images.shape
        #     image = ToPILImage()(denormalize(images))
        # else:
        #     h, w = images.shape
        #     image = ToPILImage()(images)
        # axes[1].imshow(image)
        # axes[1].axis('off')
        # axes[1].set_title(f"after_transfrom_{h}x{w}", fontsize=10)
        # plt.tight_layout()
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        # plt.close()
        # # end
        return images, image_path

def main():
    args = parse_args()
    
    # multi-gpu
    local_rank = args.local_rank
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    if local_rank == 0:
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        # print(local_rank, device)
        
    
    # output dir
    args.model_name = args.model_path.split('/')[-1]
    args.output_folder = os.path.join(args.output_folder, "images")
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    if local_rank == 0:
        print(f"Features will be saved to {args.output_folder}")

    # model
    model, processor = get_model(args.model_path)
    if local_rank == 0: print(processor)
    model = model.to(device)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # dataset
    dataset = ImageDataset(image_folder=args.image_folder, transform=processor)
    if local_rank == 0:
        print(len(dataset))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, num_workers=16, batch_size=args.batch_size, sampler=sampler)
    
    # extract features
    for batch_inputs, batch_paths in tqdm(dataloader):
        # batch_inputs = batch_inputs.to(device)
        # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #     features = get_feature_outputs(args.model_path, model.module, batch_inputs, dtype=torch.float16)
        # else:
        #     features = get_feature_outputs(args.model_path, model, batch_inputs, dtype=torch.float16)
        # assert len(features) == len(batch_paths)
        # save
        for i, image_path in enumerate(batch_paths):
            filename,_ = os.path.splitext(image_path.replace(args.image_folder, args.output_folder))
            output_path = f"{filename}.pt"
            # output_path = f"{filename}.npy"
            
            output_folder = os.path.dirname(output_path)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            
            torch.save(batch_inputs[i].to(torch.float16).clone(), output_path)
            # np.save(output_path, features[i])
            # np.savez_compressed(output_path, features[i])
            # np.savetxt(output_path, features[i], delimiter=',', fmt='%.6f')
            # with h5py.File(output_path, "w") as f:
            #     f.create_dataset("features", data=features[i], compression="gzip")

if __name__ == "__main__":
    main()
