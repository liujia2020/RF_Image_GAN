from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class StratifiedCategoryBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that keeps all requested categories present in each batch.

    The sampler expects a dataset with a ``categories`` attribute, matching the
    current RFCachedDataset. For each yielded batch, it draws
    ``samples_per_category`` indices per category. With three categories and
    samples_per_category=1, batch size is 3.
    """

    def __init__(
        self,
        categories: Sequence[str],
        required_categories: Sequence[str] = ("carotid", "muscle", "phantom"),
        samples_per_category: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 20260605,
    ):
        if samples_per_category < 1:
            raise ValueError("samples_per_category must be >= 1")

        self.categories = [str(c) for c in categories]
        self.required_categories = [str(c) for c in required_categories]
        self.samples_per_category = int(samples_per_category)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, cat in enumerate(self.categories):
            grouped[cat].append(idx)
        self.grouped = {cat: grouped.get(cat, []) for cat in self.required_categories}

        missing = [cat for cat, idxs in self.grouped.items() if not idxs]
        if missing:
            raise ValueError(f"Missing categories in sampler: {missing}")

        min_count = min(len(idxs) for idxs in self.grouped.values())
        if self.drop_last:
            self._num_batches = min_count // self.samples_per_category
        else:
            self._num_batches = math.ceil(min_count / self.samples_per_category)

    @property
    def batch_size(self) -> int:
        return len(self.required_categories) * self.samples_per_category

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        per_cat = {}
        for cat, idxs in self.grouped.items():
            values = list(idxs)
            if self.shuffle:
                rng.shuffle(values)
            per_cat[cat] = values

        for batch_idx in range(self._num_batches):
            batch: list[int] = []
            for cat in self.required_categories:
                start = batch_idx * self.samples_per_category
                end = start + self.samples_per_category
                chunk = per_cat[cat][start:end]
                if len(chunk) < self.samples_per_category:
                    if self.drop_last:
                        return
                    needed = self.samples_per_category - len(chunk)
                    chunk = chunk + per_cat[cat][:needed]
                batch.extend(chunk)
            if self.shuffle:
                rng.shuffle(batch)
            yield batch
