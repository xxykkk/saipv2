import torch
from torchvision import transforms
from PIL import Image

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