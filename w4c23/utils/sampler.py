"""
Sampling strategies for the samples in the dataset
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


from w4c23.utils.data_utils import get_file


def get_sat_sequence(sat_vals, in_seq, out_seq, min_prob):
    """Average over all timesteps the saturated values for each sequence"""
    time_vals = []
    for i in in_seq:
        time_vals.append(sat_vals[i])
    for i in out_seq:
        time_vals.append(sat_vals[i])
    val = np.mean(time_vals)
    # If sat_val is too low, do not include
    if val < 0.1:
        return -min_prob
    # If sat_val is large, set to 1 (consider +min_rob next)
    elif val > 0.6:
        return 1 - min_prob
    return val


def importance_sampling(
    samples,
    size,
    stats_path,
    radar_ds,
    preprocess_target,
    product="RATE",
    min_prob=1e-6,  # 2e-4,
):
    """
    Sample patches with proposal from Deep Generative Models of Radar
    https://arxiv.org/pdf/2104.00954.pdf
    Sampling based on amount of precipitation in the whole sequence.
    Note that original implementation in paper uses mm/h, full patch and other parameters.
    """
    if size > len(samples):
        return samples

    samples = pd.DataFrame(samples, columns=["input", "output", "region"])

    # Get amount of rain for each radar observation
    if stats_path is not None and Path(stats_path).is_file():
        with open(stats_path, "rb") as f:
            avg_saturations = pickle.load(f)
    else:
        avg_saturations = {}
        for region in radar_ds:
            region_saturations = []
            for year in radar_ds[region]:
                for sample_idx in tqdm(range(len(radar_ds[region][year]))):
                    # Get values and mask
                    values, mask = get_file(
                        sample_idx, region, product, [], preprocess_target, radar_ds
                    )
                    # Compute average of saturated values for the given timestep
                    values[mask] = 0
                    avg_sat = np.mean(1 - np.exp(-values))
                    region_saturations.append(avg_sat)
            avg_saturations[region] = region_saturations
        if stats_path is not None:
            with open(stats_path, "wb") as f:
                pickle.dump(avg_saturations, f)

    # Add sat_val to each sequence of the samples
    samples["sat_val"] = samples.apply(
        lambda x: get_sat_sequence(
            avg_saturations[x["region"]], x["input"], x["output"], min_prob
        ),
        axis=1,
    )

    # Compute weights
    weights = min_prob + samples.sat_val
    weights = weights.clip(upper=1)
    samples = samples.sample(n=size, weights=weights)

    # Remove sat vals and convert back to list
    samples = samples.drop(columns=["sat_val"])
    samples = samples.values.tolist()
    return samples
