import torch
from torch.utils.data import DistributedSampler, DataLoader
import random

class MultiHumanBatchSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        sampler, # 分给每张卡的index
        drop_last # batchsize不足的情况下是否丢掉最后一个
    ) -> None:
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.drop_last = drop_last
        self.resolution = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        self.batch_size = [56, 40,  32,  24,  24,  20,  16,  12,  12,  8,  4]

    def __iter__(self):
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        idx = random.randint(0, len(self.batch_size) - 1)
        batch_size = self.batch_size[idx]
        resolution = self.resolution[idx]
        self.sampler.dataset.set_resolution(resolution) # todo
        
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]