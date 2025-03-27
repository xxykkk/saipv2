import torch
import numpy as np
from typing import List
from vision_transformer import expert_sapiens_0_3b, expert_sapiens_1b
import torchvision.transforms as transforms
from preprocessing_theia.transfroms import ResizeAndPadToAspectRatio
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os

def get_sapiens_feature(model, processor, images, requires_grad=False, device='cpu'):
    inputs_list = []
    for i, image in enumerate(images):
        image_pil = Image.fromarray(image) # [H,W,C]->[W,H]
        image_pro = processor(image_pil).to(device)
        inputs_list.append(image_pro)
        # 可视化
        # save_dir = 'temp/feature_extraction/theia_pad'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # image_pro_np = image_pro.cpu().clone()  # 复制 tensor 到 CPU
        # image_pro_np = image_pro_np.permute(1, 2, 0)  # 将维度从 (C, H, W) 转换为 (H, W, C)
        # image_pro_np = image_pro_np.numpy()  # 转换为 NumPy 数组

        # # Denormalize the image (reverse the normalization)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_pro_np = image_pro_np * std + mean  # Denormalization

        # # Clip the values to be between 0 and 1
        # image_pro_np = np.clip(image_pro_np, 0, 1)

        # # 使用 matplotlib 绘制原始图像和处理后的图像
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # # 原始图像
        # w, h = image_pil.size
        # axes[0].imshow(image_pil)
        # axes[0].set_title(f"Original Image {h}x{w}")
        # axes[0].axis('off')

        # # 处理后的图像
        # h, w, _ = image_pro_np.shape
        # axes[1].imshow(image_pro_np)
        # axes[1].set_title(f"Processed Image {h}x{w}")
        # axes[1].axis('off')

        # # 保存图像
        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/{i}.png")
        # plt.show()
        # END
    inputs = torch.stack(inputs_list)
    if requires_grad:
        outputs, atten = model.forward_backbone(inputs)
        h,w = model.patch_resolution
        # outputs = outputs.detach().cpu().to(dtype).contiguous()
    else:
        with torch.no_grad():
            outputs, atten = model.forward_backbone(inputs)
            # h, w = model.patch_resolution
            # b,_,dim = outputs.shape
            # outputs = outputs.reshape(b,h,w,dim)
            # outputs = outputs.permute(0,3,1,2)
            # th, tw = int(h//2), int(w//2)
            # outputs = F.interpolate(outputs,size=(th,tw), mode='bilinear')
            # outputs = outputs.permute(0,2,3,1)
            # outputs = outputs.detach().cpu().to(dtype).contiguous()
    return outputs

def get_sapiens_model(model_name, target_size, device="cpu"):
	# processor = transforms.Compose([
    #     transforms.RandomResizedCrop(size=(1024,768), scale=(1,1), interpolation=3, ratio=(0.75,0.75)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ])
    processor = transforms.Compose([
        ResizeAndPadToAspectRatio(target_aspect_ratio=0.75, target_size=target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if 'sapiens_0.3b_coco_best_coco_AP_epoch_98_512x384' in model_name:
        model = expert_sapiens_0_3b(pretrained=model_name).to(device)
    elif 'sapiens_0.3b_coco_best_coco_AP_796' in model_name:
        model = expert_sapiens_0_3b(pretrained=model_name).to(device)
    elif 'sapiens_1b' in model_name:
        model = expert_sapiens_1b(pretrained=model_name).to(device)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    print("="*100, "\n", "PLZ CHECK: Load from", model_name,"\n", "="*100)
    return model, processor

def example(model, inputs, requires_grad=False):
    if requires_grad:
        outputs, atten = model.forward_backbone(inputs)
    else:
        with torch.no_grad():
            outputs, atten = model.forward_backbone(inputs)
    cls_token = outputs.last_hidden_state[:, :1]  # (1, 1, 1024) if vit-large
    visual_tokens = outputs.last_hidden_state[:, 1:]  # (1, 256, 1024) if vit-large
    pooled_cls_token = outputs.pooler_output.unsqueeze(1)  # (1, 1, 1024) if vit-large
    batch_size, num_patches, num_channels = visual_tokens.size()
    visual_tokens = visual_tokens.transpose(1, 2)
    visual_tokens = visual_tokens.reshape(
        batch_size, num_channels, int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
    )  # (1, 1024, 16, 16) BCHW for vit-huge
    return cls_token, visual_tokens, pooled_cls_token