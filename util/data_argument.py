import os

import PIL.Image
from torchvision import transforms
from util import misc
import random
import PIL
import torch.nn.functional as F
import torch

class DataAugmentationV1(object):
    def __init__(self, size, crop_size, global_crops_scale, local_crops_scale, local_crops_number, ref_size=(256, 256)):
        ref_oimage_path='/mnt/hdd4/zhangshuai/data/pretrain_dataset/train/others'
        self.list_ref_oimg_files = []
        for file in os.listdir(ref_oimage_path):
            file_path = os.path.join(ref_oimage_path, file)
            self.list_ref_oimg_files.append(file_path)
        self.num_ref_obj = len(self.list_ref_oimg_files)

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
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
        ratio = (0.4,0.6)
        if size == (224,224):
            ratio = (0.75, 1.3333333333333333)
        elif size == (256,192):
            ratio = (0.4,0.6)
        elif size == (256,128):
            ratio = (0.4,0.6)
        elif size == (384,128):
            ratio = (0.25,0.4)
        print(global_crops_scale, size, ratio)

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),  # 3 is bicubic
            # transforms.RandomHorizontalFlip(),
            flip_and_color_jitter,
            misc.GaussianBlur(1.0),
            normalize]
        )
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(0.1),
            misc.Solarization(0.2),
            normalize,
        ])
        self.global_transfo3 = transforms.Compose([
            transforms.RandomResizedCrop(size=ref_size, scale=(0.2, 1.0), interpolation=3, ratio=(0.75, 1.3333333333333333)),
            flip_and_color_jitter,
            misc.GaussianBlur(0.1),
            misc.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=local_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(p=0.5),
            normalize,
        ])
        
        
        assert crop_size[0] < ref_size[0] and crop_size[1] < ref_size[1]
        self.end_points = (ref_size[0]-crop_size[0]-1, ref_size[1]-crop_size[1]-1)
        self.region_size = crop_size

    def __call__(self, image):

        multi_scales = []
        aug_img_1 = self.global_transform(image)
        # aug_img_2 = torch.nn.functional.interpolate(aug_img_1.unsqueeze(0), scale_factor=2).squeeze(0)
        aug_img_2 = self.global_transfo2(image)
        multi_scales.append(aug_img_1)
        multi_scales.append(aug_img_2)

        for _ in range(self.local_crops_number):
            multi_scales.append(self.local_transfo(image))
        # region_img = torch.nn.functional.interpolate(aug_img_1.unsqueeze(0), size=self.region_size).squeeze(0)
        # multi_scales.append(region_img)
        # random.seed(time.time())
        # idx_obj = random.randint(0,self.num_ref_obj-1)
        # obj_img = Image.open(self.list_ref_oimg_files[idx_obj]).convert('RGB')
        # obj_img = self.global_transfo3(obj_img)

        # start_point_h = int(np.random.choice(np.arange(0, self.end_points[0], 1), 1))
        # start_point_w = int(np.random.choice(np.arange(0, self.end_points[1] - self.region_size[1], 1), 1))
        # end_point_h = start_point_h + self.region_size[0]
        # end_point_w = start_point_w + self.region_size[1]
        # obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w] = 0.7*multi_scales[-2] + 0.3*obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w]
        # roi_mask = torch.zeros_like(obj_img)[0]
        # roi_mask[start_point_h:end_point_h, start_point_w:end_point_w] += 1.

        # start_point_h = int(np.random.choice(np.arange(0, self.end_points[0], 1), 1))
        # start_point_w = int(np.random.choice(np.arange(end_point_w, self.end_points[1], 1), 1))
        # end_point_h = start_point_h + self.region_size[0]
        # end_point_w = start_point_w + self.region_size[1]
        # obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w] = 0.7*multi_scales[-1] + 0.3*obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w]
 
        # inst_mask = torch.zeros_like(obj_img)[0]
        # inst_mask[start_point_h:end_point_h, start_point_w:end_point_w] += 1.

        # multi_scales.append(obj_img)
        # multi_scales.append(inst_mask)
        # multi_scales.append(roi_mask)

        return multi_scales
    
class DataAugmentation(object):
    def __init__(self, teacher_size, size, crop_size, global_crops_scale, local_crops_scale, local_crops_number):
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
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
        ratio = (0.4,0.6)
        if size == (224,224):
            ratio = (0.75, 1.3333333333333333)
        elif size == (256,192):
            ratio = (0.4,0.6)
        elif size == (256,128):
            ratio = (0.4,0.6)
        elif size == (384,128):
            ratio = (0.25,0.4)
        print(teacher_size, global_crops_scale, size, ratio)

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=teacher_size, scale=global_crops_scale, interpolation=3, ratio=ratio),  # 3 is bicubic
            # transforms.RandomHorizontalFlip(),
            flip_and_color_jitter,
            misc.GaussianBlur(1.0),
            normalize]
        )
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(0.1),
            misc.Solarization(0.2),
            normalize,
        ])

        self.local_crops_number = local_crops_number

        self.local_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=local_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(p=0.5),
            normalize,
        ])

        self.local_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(0.5, 0.75), interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(p=0.5),
            normalize,
        ])


    def __call__(self, image):

        multi_scales = []
        aug_img_1 = self.global_transform(image)
        aug_img_2 = self.global_transfo2(image)
        multi_scales.append(aug_img_1)
        multi_scales.append(aug_img_2)

        for _ in range(self.local_crops_number):
            target_s = random.randint(0, 1)
            if target_s==0:
                multi_scales.append(self.local_transfo1(image))
            elif target_s==1:
                multi_scales.append(self.local_transfo2(image))
            else:
                print('error')
       

        return multi_scales
    
class DataAugmentationv3(object):
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
        ratio = (0.4,0.6)
        if size == (224,224):
            ratio = (0.75, 1.3333333333333333)
        elif size == (256,192):
            ratio = (0.4,0.6)
        elif size == (256,128):
            ratio = (0.4,0.6)
        elif size == (384,128):
            ratio = (0.25,0.4)
        elif size == (384,288) or size == (1024,768):
            ratio = (0.75,0.75)
            global_crops_scale = (1,1)
        print(global_crops_scale, size, ratio)

        self.global_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),  # 3 is bicubic
            ResizeAndPadToAspectRatio(target_aspect_ratio=0.75, target_size=size),
            # transforms.RandomHorizontalFlip(),
            # flip_and_color_jitter,
            color_jitter,
            misc.GaussianBlur(1.0),
            normalize]
        )
        self.global_transfo2 = transforms.Compose([
            # transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),
            ResizeAndPadToAspectRatio(target_aspect_ratio=0.75, target_size=size),
            # flip_and_color_jitter,
            color_jitter,
            misc.GaussianBlur(0.1),
            misc.Solarization(0.2),
            normalize,
        ])

        self.local_crops_number = local_crops_number

        self.local_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=local_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(p=0.5),
            normalize,
        ])

        self.local_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(0.5, 0.75), interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(p=0.5),
            normalize,
        ])


    def __call__(self, image):
        image = PIL.Image.fromarray(image)
        multi_scales = []
        aug_img_1 = self.global_transform(image)
        aug_img_2 = self.global_transfo2(image)
        multi_scales.append(aug_img_1)
        multi_scales.append(aug_img_2)

        for _ in range(self.local_crops_number):
            target_s = random.randint(0, 1)
            if target_s==0:
                multi_scales.append(self.local_transfo1(image))
            elif target_s==1:
                multi_scales.append(self.local_transfo2(image))
            else:
                print('error')

        return multi_scales
    
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
            color_jitter,
            misc.GaussianBlur(1.0),
            normalize]
        )

    def __call__(self, image):
        image = PIL.Image.fromarray(image)

        aug_img_1 = self.global_transform(image)

        return [aug_img_1]
    
class ResizeAndPadToAspectRatio:
    def __init__(self, target_aspect_ratio=0.75, target_size=(1024, 768)):
        self.target_aspect_ratio = target_aspect_ratio
        self.target_size = target_size
        self.resize_transform = transforms.Resize(target_size)
    
    def __call__(self, img): # img: PIL.Image
        if isinstance(img, PIL.Image.Image):
            width, height = img.size
        elif isinstance(img, torch.Tensor):
            width, height = img.shape[2:]
        else:
            raise TypeError

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
    
def resize_tensors_aspect_ratio(tensors, short_side_length, max_size=1333):
    N, C, H, W = tensors.shape
    scale = short_side_length / min(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_h = int(round(new_h * scale))
        new_w = int(round(new_w * scale))
    
    new_size = (new_h, new_w)

    resized_tensors = F.interpolate(tensors, size=new_size, mode='bilinear', align_corners=False)
    return resized_tensors

class RandomResizedCropWrapper(object):
    def __init__(self, size, scale, interpolation, ratio=None):
        assert isinstance(size, (list, tuple))
        self.size = size
        self.global_crops_scale = scale
        self.interpolation = interpolation
        self.ratio = ratio

    def __call__(self, img):
        size = random.choice(self.size)
        if size == (224,224):
            self.ratio = (0.75, 1.3333333333333333)
        elif size == (256,192):
            self.ratio = (0.4,0.6)
        elif size == (256,128):
            self.ratio = (0.4,0.6)
        elif size == (384,128):
            self.ratio = (0.25,0.4)
        return transforms.RandomResizedCrop(size=size, scale=self.global_crops_scale, interpolation=self.interpolation, ratio=self.ratio)(img)
    
class ResizeCropWrapper(object):
    def __init__(self, size, maxsize=1333):
        assert isinstance(size, (list, tuple))
        self.size = size
        self.maxsize = maxsize

    def __call__(self, img):
        size = random.choice(self.size)
        return transforms.Resize(size=size, maxsize=self.maxsize)(img)
