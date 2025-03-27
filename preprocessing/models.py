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

def get_sapiens_feature(model, inputs, requires_grad=False, dtype: torch.dtype = torch.float16):
    if requires_grad:
        outputs, atten = model.forward_backbone(inputs)
        # h,w = model.patch_resolution
        outputs = outputs.detach().cpu().to(dtype).contiguous()
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
            outputs = outputs.detach().cpu().to(dtype).contiguous()
    return outputs

def get_sapiens_model(model_name):
	# processor = transforms.Compose([
    #     transforms.RandomResizedCrop(size=(1024,768), scale=(1,1), interpolation=3, ratio=(0.75,0.75)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ])
    processor = transforms.Compose([
        ResizeAndPadToAspectRatio(target_aspect_ratio=0.75, target_size=(1024,768)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if 'sapiens_0.3b' in model_name:
        model = expert_sapiens_0_3b(pretrained='pretrained_models/sapiens/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth')
    elif 'sapiens_1b' in model_name:
        model = expert_sapiens_1b(pretrained='pretrained_models/sapiens/sapiens_1b/sapiens_1b_coco_best_coco_AP_821.pth')
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
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