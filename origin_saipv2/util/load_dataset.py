import os
import cv2
import math
import glob
import json
import torch

import numpy as np
import torch.utils
import torch.utils.data
import webdataset as wds
import torchvision.datasets as datasets

from io import BytesIO
from PIL import Image
from functools import partial
from torch.utils.data import IterableDataset

from safetensors.torch import load as sft_load

class HD1MDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, feature_folder, transform=None):
        # ${images_folder}/
        #   |-train/
        #   |   |-aic_persons/
        #   |   |   |-xxx.jpg
        #   |   |-cihp-persons
        # ${features_folder}/
        #   |-train/
        #   |   |-aic_persons/
        #   |   |   |-xxx.npy
        #   |   |-cihp-persons
        super().__init__()
        self.image_folder = image_folder
        self.feature_folder = feature_folder # /mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_HD1M/{model_name}
        self.transform = transform
        
        if not os.path.exists(self.image_folder):
            raise ValueError(f"{self.image_folder} not exites!")
        if not os.path.exists(self.feature_folder):
            raise ValueError(f"{self.feature_folder} not exites!")
        assert self.transform is not None
        
        self.image_paths = []
        for root, dirs, files in os.walk(self.image_folder, followlinks=True):
            for f in files:
                if f.endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(root, f))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in {self.image_folder}!")
        
        self.feature_paths = [
            os.path.splitext(img.replace(self.image_folder, self.feature_folder))[0] + '.npy'
            for img in self.image_paths
        ]
        # self.fake_features = torch.randn(3072, 1024)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        feature_path = self.feature_paths[index]
        
        # Two base images (1024x768) for teacher model and student model. Non-geometrical data argument.
        # N croped images (128x96) for student and ema student.
        image = Image.open(image_path).convert('RGB')
        features = torch.tensor(np.load(feature_path))
        # print(features.shape)
        # features = self.fake_features
        samples = self.transform(image) # List
        return samples, features
    
class HD1MDatasetV2(datasets.ImageFolder):
    def __init__(self, image_folder, feature_folder, transform=None):
        # ${images_folder}/
        #   |-train/
        #   |   |-aic_persons/
        #   |   |   |-xxx.jpg
        #   |   |-cihp-persons
        # ${features_folder}/
        #   |-train/
        #   |   |-aic_persons/
        #   |   |   |-xxx.npy
        #   |   |-cihp-persons
        super().__init__(image_folder, transform=transform)
        self.feature_folder = feature_folder

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        image_path, _ = self.samples[index]
        feature_path = image_path.replace(self.root, self.feature_folder).replace('.jpg', '.npy')

        # start_time = time.time()
        features = torch.tensor(np.load(feature_path))
        # features = torch.tensor(np.load(f'temp/fake_features/sample_{index%1000}.npy', allow_pickle=False))
        # print(f"Load npy time: {str((time.time()-start_time))}")
        return image, features
    
class HD1MDatasetV3(torch.utils.data.Dataset):
    def __init__(self, image_folder, feature_folder, transform=None):
        # ${images_folder}/
        #   |-train/
        #   |   |-aic_persons/
        #   |   |   |-xxx.jpg
        #   |   |-cihp-persons
        # ${features_folder}/
        #   |-train/
        #   |   |-aic_persons/
        #   |   |   |-xxx.npy
        #   |   |-cihp-persons
        super().__init__()
        self.image_folder = f"{image_folder}"
        self.feature_folder = feature_folder # /mnt/nfs/HAG/wangxuanhan/datasets/preprocessed_HD1M/{model_name}
        self.transform = transform
        
        if not os.path.exists(self.image_folder):
            raise ValueError(f"{self.image_folder} not exites!")
        if not os.path.exists(self.feature_folder):
            raise ValueError(f"{self.feature_folder} not exites!")
        assert self.transform is not None
        
        self.image_paths = []
        # for root, dirs, files in os.walk(self.image_folder, followlinks=True):
        for root, dirs, files in os.walk(f"{self.image_folder}/train/mhpv2_persons", followlinks=True):
            for f in files:
                if f.endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(root, f))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in {self.image_folder}!")
        
        self.feature_paths = [
            os.path.splitext(img.replace(self.image_folder, self.feature_folder))[0] + '.pt'
            for img in self.image_paths
        ]
        # self.fake_features = torch.randn(3072, 1024)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        feature_path = self.feature_paths[index]

        return image_path, feature_path
    
def load_feature_stats(
    dataset_root: str, feature_models: list):
    feature_means: dict[str, torch.Tensor] = {}
    feature_vars: dict[str, torch.Tensor] = {}
    for model in feature_models:
        model_name = model.replace("/", "_")
        feature_means[model] = torch.from_numpy(np.load(os.path.join(dataset_root, f"imagenet_mean_{model_name}.npy"))).to(
            torch.bfloat16
        )
        feature_vars[model] = torch.from_numpy(np.load(os.path.join(dataset_root, f"imagenet_var_{model_name}.npy"))).to(
            torch.bfloat16
        )
    return feature_means, feature_vars

def pad_shard_paths(shard_paths: list, num_shards: int, num_parts: int) -> list:
    final_shard_paths = shard_paths
    if num_shards % num_parts != 0: # num_shards / gpus
        if num_shards < num_parts - num_shards: # num_shards 比num_parts小的情况
            for _ in range(math.floor((num_parts - num_shards) / num_shards)):
                final_shard_paths += shard_paths[:]
            final_shard_paths += shard_paths[: num_parts - len(final_shard_paths)]
        else:
            final_shard_paths += shard_paths[: num_parts - len(final_shard_paths)] # +序列从第0个元素到倒数第num_parts - len(final_shard_paths)个元素，也就是取num_parts个元素
    return final_shard_paths

def decode_sample(key: str, data: bytes, image_transform = None, feature_transform = None):
    if ".safetensors" in key: # For features
        sft = sft_load(data)
        embedding = sft["embedding"]
        if feature_transform is not None:
            embedding = feature_transform(embedding)
        if "cls_token" in sft:
            cls = sft["cls_token"]
            if feature_transform is not None:
                cls = feature_transform(cls)
                return {"embedding": embedding, "cls": cls}
        return {"embedding": embedding}
    elif key == ".image": # For raw images
        image = np.load(BytesIO(data))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image_transform is not None:
            return image_transform(image)
        return image
    else:
        return data
    
def normalize_feature(
    x: torch.Tensor, mean = None, std = None
) -> torch.Tensor:
    return x if mean is None or std is None else (x - mean) / std

class RandomMix(IterableDataset):
    """A random interleave of multiple iterable datasets."""

    def __init__(
        self,
        datasets,
        seed = 0,
    ) -> None:
        self.datasets = datasets
        self.seed = seed

    def __iter__(self):
        """Return an iterator over the sources."""
        source = iter(self.datasets[0])
        while source is not None:
            try:
                yield next(source)
            except StopIteration:
                # del sources
                # source = None
                break

def get_image_webdataset(dataset_root, 
                         feature_models, 
                         dataset, 
                         split, 
                         image_transform, 
                         feature_norm=None, 
                         seed=0, 
                         shuffle=False, 
                         world_size=1, 
                         **kwargs):
    all_feature_datasets = {}
    if feature_norm:
        feature_means, feature_vars = load_feature_stats(dataset_root, feature_models)
    with open(os.path.join(dataset_root, dataset, "splits.json"), "r") as splitf:
        dataset_len = json.load(splitf)[split]
        assert dataset_len != 0
    
    # path_pattern = os.path.join("/mnt/hdd4/wangxuanhan/datasets/preprocessed_theia_data", dataset, "images", f"*-{split}.tar") # 加载raw images
    path_pattern = os.path.join(dataset_root, dataset, "images", f"*-{split}.tar") # 加载raw images
    if "image" not in all_feature_datasets:
        all_feature_datasets["image"] = []
    shard_paths = sorted(glob.glob(path_pattern))
    num_shards = len(shard_paths)
    num_parts = world_size
    final_shard_paths = pad_shard_paths(shard_paths, num_shards, num_parts)
    ds = wds.WebDataset(
            final_shard_paths,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            detshuffle=True,
            shardshuffle=shuffle,
            seed=seed,
            cache_dir='./cache'
    ).decode(partial(decode_sample, image_transform=image_transform))
    all_feature_datasets["image"].append(ds)
    print(f"Image: {path_pattern} num_shard: {num_shards} after pad: {len(final_shard_paths)}")
    
    for model_name in feature_models:
        path_pattern = os.path.join(dataset_root, dataset, f"{model_name.replace('/', '_')}", f"*-{split}.tar")
        rename_kw = {model_name: model_name.replace("/", "_").lower() + ".safetensors"}  # replace v by k
        
        if model_name not in all_feature_datasets:
            all_feature_datasets[model_name] = []
        
        shard_paths = sorted(glob.glob(path_pattern))
        num_shards = len(shard_paths)
        num_parts = world_size
        final_shard_paths = pad_shard_paths(shard_paths, num_shards, num_parts)
        if feature_norm:
            feature_transform = partial(
                normalize_feature, mean=feature_means[model_name], std=feature_vars[model_name]
            )
        else:
            feature_transform = None
        ds = (
            wds.WebDataset(
                final_shard_paths,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                detshuffle=True,
                shardshuffle=shuffle,
                seed=seed,
                cache_dir='./cache'
            )
            .decode(partial(decode_sample, image_transform=image_transform, feature_transform=feature_transform))
            .rename(keep=True, **rename_kw)
        )
        all_feature_datasets[model_name].append(ds)
        print(f"{model_name}: {path_pattern} num_shard: {num_shards} after pad: {len(final_shard_paths)}")
    # {'images': [webdataset],
    # 'model1': [webdataset],...
    # 'modelM': [webdataset],}
    combined_feature_datasets = {}
    for feature_set_name, fds in all_feature_datasets.items():
        ds = RandomMix(fds, seed=seed)
        combined_feature_datasets[feature_set_name] = ds
    
    return combined_feature_datasets, dataset_len

def get_image_webdataloader(
    datasets: dict,
    batch_size= None,
    shuffle = False,
    shuffle_buffer_size = 1_000,
    seed = 0,
    **kwargs,
) -> dict:
    loaders = {}
    for k in datasets: # image, model1, mode2,..., modelN
        loader = wds.WebLoader(datasets[k], batch_size=None, generator=torch.default_generator, **kwargs)
        if shuffle:
            loader = loader.shuffle(shuffle_buffer_size, seed=seed)  # shuffle after mix
        loader = loader.batched(batch_size, collation_fn=torch.utils.data.default_collate)
        loaders[k] = loader
    return loaders

def get_image_iterator(data_loaders):

    packed_loader = data_loaders.get("packed", None)
    # place packed_loader at the first
    if packed_loader is not None:
        loaders = [packed_loader, *[data_loaders[k] for k in data_loaders if k != "packed"]]
    else:
        loaders = list(data_loaders.values())
        
    # iterators = [iter(loader) for loader in loaders]

    # merge dicts
    # this is to accommodate the old organization of datasets (each shard contains one or more columns,
    # and images are duplicated columns).
    # In new (current) dataset organization (columns are completely separated),
    # column keys are all different except some "built-in" keys added by webdataset,
    # but they are not related to any data, training, so on.
    # During transit from old to new, where two organizations exist at the same time,
    # this is to ignore extra "image" field in datasets loaded.
    for data in zip(*loaders):
        ## check
        keys = data[0]['__key__']
        for i, key in enumerate(keys):
            if data[1]['__key__'][i] != key:
                raise ValueError
        ## end check
        # yield data
        for i in range(1, len(loaders)):
            for k in data[i]:
                if k not in data[0]:
                    data[0][k] = data[i][k]
        yield data[0]