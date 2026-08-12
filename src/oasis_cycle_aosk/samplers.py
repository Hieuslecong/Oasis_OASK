"""Deterministic mixed-batch sampling for crack + true-normal training."""
import math

import torch
from torch.utils.data import Sampler


class MixedBatchSampler(Sampler):
    """Yield fixed-composition batches from crack and normal datasets.

    The concatenated dataset is expected to be ordered as:

        [crack dataset rows][normal dataset rows]

    Each epoch exposes every crack sample approximately once (subject to the
    final partial cycle) while cycling through the normal pool as needed. The
    composition is deterministic for a given seed and epoch.
    """

    def __init__(
        self,
        crack_count,
        normal_count,
        batch_size,
        normal_fraction,
        seed=1337,
    ):
        self.crack_count = int(crack_count)
        self.normal_count = int(normal_count)
        self.batch_size = int(batch_size)
        self.normal_fraction = float(normal_fraction)
        self.seed = int(seed)
        self.epoch = 0

        if self.crack_count <= 0:
            raise ValueError("crack_count must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not 0.0 <= self.normal_fraction < 1.0:
            raise ValueError("normal_fraction must satisfy 0 <= f < 1")

        self.normal_per_batch = int(round(self.batch_size * self.normal_fraction))
        self.normal_per_batch = min(self.normal_per_batch, self.batch_size - 1)
        self.crack_per_batch = self.batch_size - self.normal_per_batch
        if self.normal_fraction > 0 and self.normal_per_batch == 0:
            raise ValueError(
                "requested normal_fraction is too small for this batch_size; "
                "increase batch_size or normal_fraction so at least one normal "
                "sample is present per batch"
            )
        if self.normal_per_batch > 0 and self.normal_count <= 0:
            raise ValueError("normal_fraction > 0 requires normal samples")

        self.realized_normal_fraction = self.normal_per_batch / self.batch_size
        self.steps = int(math.ceil(self.crack_count / self.crack_per_batch))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    @staticmethod
    def _infinite_permutations(count, generator):
        while True:
            for idx in torch.randperm(count, generator=generator).tolist():
                yield idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        crack_iter = self._infinite_permutations(self.crack_count, g)
        normal_iter = (
            self._infinite_permutations(self.normal_count, g)
            if self.normal_per_batch
            else None
        )

        for _ in range(self.steps):
            batch = [next(crack_iter) for _ in range(self.crack_per_batch)]
            if normal_iter is not None:
                batch.extend(
                    self.crack_count + next(normal_iter)
                    for _ in range(self.normal_per_batch)
                )
            order = torch.randperm(len(batch), generator=g).tolist()
            yield [batch[i] for i in order]
