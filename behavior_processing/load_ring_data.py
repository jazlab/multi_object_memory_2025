"""Utilities for ring dataset analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "../cache/behavior_processing/ring.csv"
RADIUS = 0.35  # Radius of the ring


def load_data(
    completed: bool = True,
    correct_threshold: float = 0.2,
) -> pd.DataFrame:
    """Load ring data.

    Args:
        completed: Whether to filter only completed trials.
        correct_threshold: Threshold for correct response.

    Returns:
        df: Ring dataset dataframe.
    """
    df = pd.read_csv(DATA_PATH)

    # Filter only completed trials if necessary
    if completed:
        df = df[df["completed"]]

    # Add correct
    correct = []
    response_per_object = []
    for _, row in df.iterrows():
        target_index = row.target_object_index
        response_object = []
        for object_index in range(2):
            target_r = row[f"object_{object_index}_r"]
            target_theta = row[f"object_{object_index}_theta"]
            response_r = row.response_r
            response_theta = row.response_theta
            target_x = target_r * np.cos(target_theta)
            target_y = target_r * np.sin(target_theta)
            response_x = response_r * np.cos(response_theta)
            response_y = response_r * np.sin(response_theta)
            euclidean_error = np.sqrt(
                (target_x - response_x) ** 2 + (target_y - response_y) ** 2
            )
            response_object.append(euclidean_error < correct_threshold)
        response_per_object.append(response_object)
        correct.append(response_object[target_index])
    df["correct"] = correct
    df["response_object_0"] = [x[0] for x in response_per_object]
    df["response_object_1"] = [x[1] for x in response_per_object]

    # Add target identity and nontarget identity
    df["target_id"] = df.apply(
        lambda x: x[f"object_{x.target_object_index}_id"],
        axis=1,
    )
    df["nontarget_id"] = df.apply(
        lambda x: x[f"object_{1 - x.target_object_index}_id"],
        axis=1,
    )

    # Add reaction time
    df["reaction_time"] = df["response_time"] - df["time_cue_onset"]

    return df
