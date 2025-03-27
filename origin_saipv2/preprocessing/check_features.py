import numpy as np
import os

image_folder = 'data/HD1M'
feature_folder = '/mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_HD1M/sapiens_0.3b_coco_best_coco_AP_796'

image_paths = []
for root, dirs, files in os.walk(image_folder, followlinks=True):
  for f in files:
      if f.endswith(('jpg', 'jpeg', 'png')):
          image_paths.append(os.path.join(root, f))

if len(image_paths) == 0:
  raise ValueError(f"No image files found in {image_folder}")

feature_paths = [image_path.replace(image_folder, feature_folder).replace('jpg','npy').replace('png','npy') for image_path in image_paths]

num_images = len(image_paths)
num_features = 0
for feature_path in feature_paths:
  if not os.path.exists(feature_path):
    pass
  else:
    num_features += 1
    test_feature = np.load(feature_path)
    print(test_feature.min(), test_feature.max())
    print(test_feature.shape)
  print(f'Images: {num_images}, Features: {num_features:07d}.', end='\r')
print(f'Images: {num_images}, Features: {num_features:07d}.')