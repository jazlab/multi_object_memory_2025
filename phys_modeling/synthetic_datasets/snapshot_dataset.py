"""Function to get snapshot dataset."""

import numpy as np
import torch


def get_snapshot_dataset(dataset, snapshot_data_dir):
    # Load snapshot data
    spikes = np.load(f"{snapshot_data_dir}/spikes.npy")
    spikes[np.isnan(spikes)] = 0.0

    # Override dataset attributes
    dataset.data["neural"] = spikes
    dataset.train_data["neural"] = spikes
    dataset.test_data["neural"] = spikes
    dataset._train_data_torch["neural"] = torch.from_numpy(
        spikes.astype(np.float32)
    )
    dataset._test_data_torch["neural"] = torch.from_numpy(
        spikes.astype(np.float32)
    )

    return dataset
