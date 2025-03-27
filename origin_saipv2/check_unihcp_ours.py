from functools import partial
import PIL.Image
import torch.nn as nn
import PIL
import os
import torch
import torchvision.transforms as transforms
from vision_transformer_no_cls import expert_unihcp,VisionTransformer
os.environ["CUDA_VISIBLE_DEVICES"]="4"

def compute_l1_distance(A, B):
    return torch.sum(torch.abs(A - B))

def compute_l2_distance(A, B):
    return torch.sqrt(torch.sum((A - B) ** 2))

checkpoint_path = "pretrained_models/unihcp/ckpt_task0_iter_newest.pth.tar"
unihcp = expert_unihcp(pretrained=checkpoint_path)

vit_base = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=0.1, use_abs_pos_emb=True,
        img_size=1344, use_cls_token=False)
checkpoint_model = torch.load(checkpoint_path, map_location='cpu')
if 'model' in checkpoint_model:
    param_dict = checkpoint_model['model']
elif 'state_dict' in checkpoint_model:
    param_dict = checkpoint_model['state_dict']
elif 'student' in checkpoint_model: ### for dino
    print('load from student')
    param_dict = checkpoint_model["student"]
else:
    param_dict = checkpoint_model
new_param_dict = {}
for k, v in param_dict.items():
    if "decoder_module" in k:
        continue
    if "neck_module" in k:
        continue
    new_key = k.replace("module.backbone_module.", "")
    new_param_dict[new_key] = v
msg = vit_base.load_state_dict(new_param_dict, strict=False)
print('Load from {}: {}'.format(checkpoint_path, msg))
del checkpoint_model, param_dict

transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
image = PIL.Image.open("11022_1102202.jpg")
input = transform(image).unsqueeze(0)

unihcp.cuda()
vit_base.cuda()
input = input.cuda()
unihcp.eval()
vit_base.eval()
unihcp_output, _ = unihcp.forward_features(input) # [1, 196, 768]
vit_base_output, _ = vit_base(input) # [1, 196, 768]

l1_distance = compute_l1_distance(unihcp_output, vit_base_output)
l2_distance = compute_l2_distance(unihcp_output, vit_base_output)
print("L1 Distance:", l1_distance.item())
print("L2 Distance:", l2_distance.item())