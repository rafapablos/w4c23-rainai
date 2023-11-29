"""
Constant definition for buckets with mean and max for each one to be used in classification problem
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Bucket:
    idx: int
    mean: float
    max: float
    weight: float


@dataclass
class BucketConstants:
    buckets: List[Bucket]
    means: List[float]
    weights: List[float]
    boundaries: List[float]
    ranges: List[float]
    num_buckets: int


# Custom buckets used for classification when using mm/h
_buckets_mmh = [
    Bucket(idx=0, mean=0, max=0.08, weight=0.5107),
    Bucket(idx=1, mean=0.12, max=0.16, weight=0.6014),
    Bucket(idx=2, mean=0.2, max=0.25, weight=0.627),
    Bucket(idx=3, mean=0.32, max=0.4, weight=0.6295),
    Bucket(idx=4, mean=0.51, max=0.63, weight=0.631),
    Bucket(idx=5, mean=0.81, max=1, weight=0.6359),
    Bucket(idx=6, mean=1.3, max=1.6, weight=0.6472),
    Bucket(idx=7, mean=2.0, max=2.5, weight=0.6667),
    Bucket(idx=8, mean=3.25, max=4, weight=0.6901),
    Bucket(idx=9, mean=5.15, max=6.3, weight=0.7298),
    Bucket(idx=10, mean=8.1, max=10, weight=0.7823),
    Bucket(idx=11, mean=13, max=16, weight=0.8428),
    Bucket(idx=12, mean=20.5, max=25, weight=0.9084),
    Bucket(idx=13, mean=32.5, max=40, weight=0.9617),
    Bucket(
        idx=14, mean=45, max=128, weight=1.0
    ),  # Max is 128 as defined by preprocessing
]

_buckets_w4c23_1 = [
    Bucket(idx=0, mean=0.1, max=0.2, weight=1.0),
    Bucket(idx=1, mean=0.6, max=1, weight=1.0),
    Bucket(idx=2, mean=3, max=5, weight=1.0),
    Bucket(idx=3, mean=7.5, max=10, weight=1.0),
    Bucket(idx=4, mean=12.5, max=15, weight=1.0),
    Bucket(idx=4, mean=20, max=128, weight=1.0),
]


# Consider adding weights?
_buckets_w4c23_2 = [
    Bucket(idx=0, mean=0, max=0.05, weight=1.0),
    Bucket(idx=1, mean=0.07, max=0.10, weight=1.0),
    Bucket(idx=2, mean=0.12, max=0.15, weight=1.0),
    Bucket(idx=3, mean=0.17, max=0.20, weight=1.0),
    Bucket(idx=4, mean=0.3, max=0.4, weight=1.0),
    Bucket(idx=5, mean=0.5, max=0.6, weight=1.0),
    Bucket(idx=6, mean=0.7, max=0.8, weight=1.0),
    Bucket(idx=7, mean=0.9, max=1.0, weight=1.0),
    Bucket(idx=8, mean=1.0, max=1.5, weight=1.0),
    Bucket(idx=9, mean=1.5, max=2.0, weight=1.0),
    Bucket(idx=10, mean=2.0, max=2.5, weight=1.0),
    Bucket(idx=11, mean=2.5, max=3.0, weight=1.0),
    Bucket(idx=12, mean=3.0, max=3.5, weight=1.0),
    Bucket(idx=13, mean=3.5, max=4.0, weight=1.0),
    Bucket(idx=14, mean=4.0, max=4.5, weight=1.0),
    Bucket(idx=15, mean=4.5, max=5.0, weight=1.0),
    Bucket(idx=16, mean=5.0, max=5.5, weight=1.0),
    Bucket(idx=17, mean=5.5, max=6.0, weight=1.0),
    Bucket(idx=18, mean=6.0, max=6.5, weight=1.0),
    Bucket(idx=19, mean=6.5, max=7.0, weight=1.0),
    Bucket(idx=20, mean=7.0, max=7.5, weight=1.0),
    Bucket(idx=21, mean=7.5, max=8.0, weight=1.0),
    Bucket(idx=22, mean=8.0, max=8.5, weight=1.0),
    Bucket(idx=23, mean=8.5, max=9.0, weight=1.0),
    Bucket(idx=24, mean=9.0, max=9.5, weight=1.0),
    Bucket(idx=25, mean=9.5, max=10.0, weight=1.0),
    Bucket(idx=26, mean=10.0, max=10.5, weight=1.0),
    Bucket(idx=27, mean=10.5, max=11.0, weight=1.0),
    Bucket(idx=28, mean=11.0, max=11.5, weight=1.0),
    Bucket(idx=29, mean=11.5, max=12.0, weight=1.0),
    Bucket(idx=30, mean=12.0, max=12.5, weight=1.0),
    Bucket(idx=31, mean=12.5, max=13.0, weight=1.0),
    Bucket(idx=32, mean=13.0, max=13.5, weight=1.0),
    Bucket(idx=33, mean=13.5, max=14.0, weight=1.0),
    Bucket(idx=34, mean=14.0, max=14.5, weight=1.0),
    Bucket(idx=35, mean=14.5, max=15.0, weight=1.0),
    Bucket(idx=36, mean=15.0, max=15.5, weight=1.0),
    Bucket(idx=37, mean=15.5, max=16.0, weight=1.0),
    Bucket(idx=38, mean=16.0, max=16.5, weight=1.0),
    Bucket(idx=39, mean=16.5, max=17.0, weight=1.0),
    Bucket(idx=40, mean=17.0, max=17.5, weight=1.0),
    Bucket(idx=41, mean=17.5, max=18.0, weight=1.0),
    Bucket(idx=42, mean=18.0, max=18.5, weight=1.0),
    Bucket(idx=43, mean=18.5, max=19.0, weight=1.0),
    Bucket(idx=44, mean=19.0, max=19.5, weight=1.0),
    Bucket(idx=45, mean=19.5, max=20.0, weight=1.0),
]

_buckets_test = [
    Bucket(idx=0, mean=0, max=0.08, weight=1.0),
    Bucket(idx=14, mean=45, max=128, weight=1.0),
]


def getBucketObject(buckets_list):
    return BucketConstants(
        buckets=buckets_list,
        means=[b.mean for b in buckets_list],
        weights=[b.weight for b in buckets_list],
        boundaries=[b.max for b in buckets_list[:-1]],
        ranges=[
            buckets_list[i].max - buckets_list[i - 1].max
            if i > 0
            else buckets_list[i].max
            for i in range(len(buckets_list))
        ],
        num_buckets=len(buckets_list),
    )


BUCKET_CONSTANTS = {
    "mmh": getBucketObject(_buckets_mmh),
    "test": getBucketObject(_buckets_test),
    "w4c23_1": getBucketObject(_buckets_w4c23_1),
    "w4c23_2": getBucketObject(_buckets_w4c23_2),
}
