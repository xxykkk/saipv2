# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import sys
sys.path.append('/mnt/hdd3/wangxuanhan/research/vfm_research/learning_research/saipv2')
import argparse
import gc
import glob
import json
import math
import multiprocessing
import os
from io import BytesIO
from os.path import join
from typing import Any, Generator, Iterable, Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
import webdataset as wds
from numpy.typing import NDArray
from safetensors.torch import save as safe_torch_save

try:
    import tensorflow_datasets as tfds
    from tensorflow.python.ops.numpy_ops import np_config
except ImportError as e:
    print (e)
    print ("No TF usable. It's ok if you are not processing OXE dataset.")
from preprocessing_theia.webdataset_utils import check_existing_shard, decode_image_npy_only
from preprocessing_theia.utils import get_feature_outputs, get_model
from torch.utils.data import IterableDataset


def get_dataset(dataset_name: str, split: str, dataset_root: Optional[str] = None) -> Tuple[Iterable, List[str]]:
    """Get the dataset and its subset keys (if has) given a dataset name.

    Args:
        dataset_name (str): name of the dataset.
        split (str): split of the dataset.
        dataset_root (Optional[str]): root dir of the dataset, if the dataset is stored locally.
            Defaults to None (remote dataset).

    Returns:
        tuple[Iterable, List[str]]: dataset and its subset keys
    """
    if dataset_root is None:
        raise ValueError("`dataset_root` is not given.")
    dataset_dir = os.path.join(dataset_root, dataset_name, "images")
    if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
        raise ValueError(f"{dataset_dir} is not found or is not a directory.")
    print("dataset shards", sorted(glob.glob(f"{dataset_dir}/*-{split}.tar"))[:5])
    dataset = wds.WebDataset(
        sorted(glob.glob(f"{dataset_dir}/*-{split}.tar")),
        shardshuffle=False,
    ).decode(decode_image_npy_only)
    return dataset, ["__self__"]



def get_episode(ds: Any) -> Generator[Tuple[Any, int], Any, Any]:
    """Get an episode / a trajectory / a segment form the dataset

    Args:
        ds (Any): oxe dataset in tfds format or image/video dataset in webdataset format.

    Yields:
        Generator[tuple[Any, int], Any, Any]: a trajectory with its length.
    """
    if isinstance(ds, IterableDataset):
        it = iter(ds)
        while True:
            sample_buff = []
            try:
                for _ in range(1000):
                    sample = next(it)
                    sample_buff.append(sample)
                yield sample_buff, len(sample_buff)
            except StopIteration:
                yield sample_buff, len(sample_buff)
                break
    else:
        for ep in ds:
            yield ep, len(ep["steps"])


def get_images(ep: Any, subset: str) -> Tuple[List[NDArray], Optional[List[str]]]:
    """Get images from an episode / a trajectory.

    Args:
        ep (Any): an episode / a trajectory.
        subset (str): subset name.

    Returns:
        tuple[List[NDArray], Optional[List[str]]]: extracted images with optional info.
    """
    if isinstance(ep, list):  # for image / video dataset, no subsets
        return [step["image"] for step in ep], [step["__key__"] for step in ep]
    else:  # for oxe dataset, subset means multiple camera views
        images: list[NDArray] = []
        for step in ep["steps"]:
            image = cv2.resize(step["observation"][subset].numpy(), (224, 224))
            images.append(image)
        return images, None


def get_shard_dir(root: str, subset: str, key: str) -> str:
    """Get the directory to hold shards.

    Args:
        root (str): root directory.
        subset (str): subset name.
        key (str): key (column) name of the processed dataset. Usually it is the name of the feature / input.

    Returns:
        str: directory to hold the shards.
    """
    if subset == "__self__":
        return os.path.join(root, key)
    else:
        return os.path.join(root, subset, key)


def get_shard_filename(dataset_name: str, subset: str, split: str, shard_idx: int) -> str:
    """Get file name of the shard.

    Args:
        dataset_name (str): name of the dataset.
        subset (str): name of the subset.
        split (str): name of the split.
        shard_idx (int): index of this shard.

    Returns:
        str: shard file name.
    """
    if subset == "__self__":
        return f"{dataset_name}_{split}-{shard_idx:06d}-{split}.tar"
    else:
        return f"{dataset_name}_{subset}_{split}-{shard_idx:06d}-{split}.tar"


def feature_extractor(
    args: argparse.Namespace,
    shard_queue: multiprocessing.Queue, # 几个shard
    worker_id: int, # 第几张卡
    dataset_len: int = 0, # 数据集总的有多少，HD1M 1218136
) -> None:
    """Feature extractor, operating on each `worker_id`.

    Args:
        args (argparse.Namespace): configurations.
        shard_queue (multiprocessing.Queue): queue to get shard index to work on.
        worker_id (int): id of this worker.
        dataset_len (int): length of the entire dataset to be processed.
    """
    if args.model != "image":
        model, processor = get_model(args.model, device=worker_id)
    else:
        model, processor = None, None
    print(processor)
    dataset, subsets = get_dataset(args.dataset, args.split, args.dataset_root)
    for sample in dataset:
        for key, value in sample.items():  # 遍历样本中的所有键值对
            if isinstance(value, np.ndarray):  # 如果值是 NumPy 数组
                print(f"Key: {key}, Shape: {value.shape}")  # 打印形状
            else:
                print(f"Key: {key}, Value: {value}")  # 直接打印值
        break
    dataset_output_root = join(args.output_path, args.dataset)

    cum_traj_len, traj_index = 0, 0
    shard_idx = shard_queue.get()
    data_iter = get_episode(dataset) # 封装一下，是一个迭代对象，每次返回1000条数据
    episode, traj_len = next(data_iter)
    remain_traj_len = traj_len
    while shard_idx is not None: # 对于本进程要处理的shard
        print(f"{args.dataset} {args.model} shard {shard_idx:04d} worker {worker_id} " f"Subsets: {subsets}")
        # navigate (stream) the dataset to the correct trajectory
        while (cum_traj_len + remain_traj_len) <= shard_idx * args.samples_per_shard:
            cum_traj_len += remain_traj_len
            try:
                episode, traj_len = next(data_iter) # 取1000个数据，返回sample_buff: List和len(sample_buff)
                remain_traj_len = traj_len
                traj_index += 1
            except StopIteration:
                break

        # check shard
        # model_names_legit = args.model.replace("/", "_").replace("models_","")
        model_names_legit = os.path.splitext(os.path.basename(args.model))[0]
        shard_keys = [model_names_legit]
        subset_check_codes = {subset: {k: 0 for k in shard_keys} for subset in subsets} # {'__self__': {'sapiens_0.3b_coco_best_coco_AP_796': 0}}
        print("="*100, "\n", "PLZ CHECK", subset_check_codes,"\n", "="*100)

        for subset in subsets: # 对于每个数据集
            for k in shard_keys: # 对于当前输入的要提取特征的每个模型
                shard_dir = get_shard_dir(dataset_output_root, subset, k)
                shard_filename = get_shard_filename(args.dataset, subset, args.split, shard_idx)
                shard_path = os.path.join(shard_dir, shard_filename)
                shard_check_code, _ = check_existing_shard(shard_path, shard_keys) # 检查已经提好了特征
                subset_check_codes[subset][k] = shard_check_code

        # generate data to the shard buffers
        subset_shard_buffers: Dict[str, Dict[str, List[Dict[str, str | bytes]]]] = {
            subset: {k: [] for k in shard_keys} for subset in subsets
        }
        while cum_traj_len < min((shard_idx + 1) * args.samples_per_shard, dataset_len): # 对于每一条数据
            for subset in subsets: # 对于每一个数据集
                images, info = None, None

                start_frame_index = traj_len - remain_traj_len
                if start_frame_index >= traj_len:
                    raise ValueError("calculate start frame index error, needs more trajectories")
                # end of the trajectory
                end_frame_index = min((shard_idx + 1) * args.samples_per_shard - cum_traj_len, traj_len)

                # generate shard data per key, including images and model features
                # skip any indices that are completed
                for k in shard_keys: # 对于每一个模型
                    if subset_check_codes[subset][k] == 1:
                        print(f"{args.dataset} {subset} {k} shard {shard_idx:04d} check pass")
                        continue
                    if k == "image":
                        if images is None:
                            # read all the images in the trejectory
                            images, info = get_images(episode, subset)
                        for frame_index in range(start_frame_index, end_frame_index):
                            basename = info[frame_index] if info else ""
                            if not args.dry_run:
                                image_out = BytesIO()
                                np.save(image_out, images[frame_index])
                                subset_shard_buffers[subset][k].append({"__key__": basename, k: image_out.getvalue()})
                    else:
                        if images is None:
                            images, info = get_images(episode, subset) # 获得image的list和__key__的list
                        processed = start_frame_index
                        # batch processing images
                        while processed < end_frame_index: # 没有处理到最后一个
                            # take a batch
                            batch_images = images[processed : processed + args.batch_size]
                            if not args.dry_run:
                                effective_batch_size = len(batch_images)
                                features = get_feature_outputs(k, model, processor, batch_images, device=worker_id)
                                for frame_index in range(processed, processed + effective_batch_size):
                                    basename = info[frame_index] if info else ""
                                    tensor_sample_buffer = {}
                                    for feature_key in features[k]: # {'sapiens_0.3b_coco_best_coco_AP_796': {'embedding':...} 对于模型k的特征
                                        tensor_sample_buffer[feature_key] = features[k][feature_key][
                                            frame_index - processed
                                        ]
                                    subset_shard_buffers[subset][k].append(
                                        {"__key__": basename, f"{k}.safetensors": safe_torch_save(tensor_sample_buffer)}
                                    )

                            # next batch
                            processed += args.batch_size

            cum_traj_len += (
                end_frame_index - start_frame_index
            )  # only increase processed traj len by the actual number of frames processed
            remain_traj_len -= end_frame_index - start_frame_index
            print(f"{args.dataset} {args.model} shard {shard_idx:04d} traj {traj_index:06d} remains {remain_traj_len}")
            # if the trajectory is exhausted, get the next one
            if remain_traj_len == 0:
                try:
                    episode, traj_len = next(data_iter)
                    remain_traj_len = traj_len
                    traj_index += 1
                except StopIteration:
                    break

        # shard_buffer generate done, write shard
        if not args.dry_run:
            for subset in subsets: # 表示多少个数据集
                for k in shard_keys: # 表示模型
                    if subset_check_codes[subset][k] == 1:
                        continue
                    shard_dir = get_shard_dir(dataset_output_root, subset, k)
                    shard_filename = get_shard_filename(args.dataset, subset, args.split, shard_idx)
                    shard_path = os.path.join(shard_dir, shard_filename)
                    if not os.path.exists(shard_dir):
                        os.makedirs(shard_dir, exist_ok=True)
                    print(len(subset_shard_buffers[subset][k]))
                    with wds.TarWriter(shard_path) as tar_writer:
                        for sample in subset_shard_buffers[subset][k]:
                            tar_writer.write(sample)

        print(f"{args.dataset} {args.model} shard {shard_idx:04d} done")
        del subset_shard_buffers
        gc.collect()
        # get a new shard to process
        shard_idx = shard_queue.get()


def main() -> None:
    """Main entry of feature extraction"""
    # np_config.enable_numpy_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset-root", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0, help="start index (form 0) of **steps** to process")
    parser.add_argument(
        "--num-to-process",
        type=int,
        default=-1,
        help="number of **steps** to process based on start. -1 means all remaining from the start.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for the model forward pass")
    parser.add_argument("--force", action="store_true", help="force overwrite existing feature files.")
    parser.add_argument("--dry-run", action="store_true", help="do not do model forward pass and write out.")
    parser.add_argument(
        "--samples-per-shard", type=int, default=1000, help="number of samples per webdataset shard. Rarely changed."
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to parallel")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", 12345)) # 指定主机和端口
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client() # 等待 VSCode 连接
        debugpy.breakpoint() # 设置断点
        print("Debugger is attached.")

    if torch.cuda.is_available():
        args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
    else:
        args.num_gpus = 0

    # make directories
    dataset_output_root = os.path.join(args.output_path, args.dataset)
    if not os.path.exists(dataset_output_root):
        os.makedirs(dataset_output_root)
    print(f'Output root: {dataset_output_root}')

    # organize the start index to start of a shard
    start_fi = args.start // args.samples_per_shard * args.samples_per_shard # 0, 1000, 1000
    start_shard_idx = start_fi // args.samples_per_shard

    dataset_dir = os.path.join(args.dataset_root, args.dataset)
    print(f'Dataset dir: {dataset_dir}')

    with open(os.path.join(dataset_dir, "splits.json"), "r") as f:
        splits = json.load(f)
        dataset_len = splits[args.split]

    # calculate how many shards to create
    if args.num_to_process > 0:
        end_sample_index = args.start + args.num_to_process
    else:
        end_sample_index = dataset_len

    if end_sample_index % args.samples_per_shard == 0: # 计算一个有多少个shard
        end_shard_idx = end_sample_index // args.samples_per_shard
    else:
        end_shard_idx = math.ceil((end_sample_index) / args.samples_per_shard)
    shards = list(range(start_shard_idx, end_shard_idx)) # 从0开始到end_shard_idx-1

    # create a queue to hold shards
    shard_queue: multiprocessing.Queue = multiprocessing.Queue()
    for shard_idx in shards:
        shard_queue.put(shard_idx)
    for _ in range(args.num_gpus * 2 + 1):
        shard_queue.put(None)

    # create workers
    workers = [
        multiprocessing.Process(target=feature_extractor, args=(args, shard_queue, worker_id, dataset_len))
        for worker_id in range(max(args.num_gpus, 1))
    ]

    for w in workers:
        w.start()
    for w in workers:
        w.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
