"""Utilities for triangle dataset analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "../cache/behavior_processing/triangle.csv"
RADIUS = 0.35  # Radius of the triangle
TRIANGLE_THETAS = [
    0,  # right
    2 * np.pi / 3,  # top-left
    4 * np.pi / 3,  # bottom-left
]
TRIANGLE_POSITIONS = [
    (RADIUS * np.cos(theta), RADIUS * np.sin(theta))
    for theta in TRIANGLE_THETAS
]


def load_data(
    only_triangle_trials: bool = True,
    completed: bool = True,
    correct_threshold: float = 0.2,
) -> pd.DataFrame:
    """Load triangle data.

    Args:
        only_triangle_trials: Whether to filter only triangle trials.
        completed: Whether to filter only completed trials.
        correct_threshold: Threshold for correct response.

    Returns:
        df: Triangle dataset dataframe.
    """
    df = pd.read_csv(DATA_PATH)

    # Filter only triangle trials if necessary
    if only_triangle_trials:
        df = df[df.on_triangle]

    # Filter only completed trials if necessary
    if completed:
        df = df[df["completed"]]

    # Add target location
    target_location = []
    for i, row in df.iterrows():
        target_index = row.target_object_index
        target_loc = row[f"object_{target_index}_location"]
        target_location.append(target_loc)
    df["target_location"] = target_location

    # Add correct
    correct = []
    for i, row in df.iterrows():
        target_index = row.target_object_index
        target_r = row[f"object_{target_index}_r"]
        target_theta = row[f"object_{target_index}_theta"]
        response_r = row.response_r
        response_theta = row.response_theta
        target_x = target_r * np.cos(target_theta)
        target_y = target_r * np.sin(target_theta)
        response_x = response_r * np.cos(response_theta)
        response_y = response_r * np.sin(response_theta)
        euclidean_error = np.sqrt(
            (target_x - response_x) ** 2 + (target_y - response_y) ** 2
        )
        correct.append(euclidean_error < correct_threshold)
    df["correct"] = correct

    # Add response location
    response_location = []
    for _, row in df.iterrows():
        response_loc = np.nan
        response_x = row.response_r * np.cos(row.response_theta)
        response_y = row.response_r * np.sin(row.response_theta)
        for i, pos in enumerate(TRIANGLE_POSITIONS):
            euclidean_error = np.sqrt(
                (pos[0] - response_x) ** 2 + (pos[1] - response_y) ** 2
            )
            if euclidean_error < correct_threshold:
                response_loc = i
                break
        response_location.append(response_loc)
    df["response_location"] = response_location

    # Add target identity and response identity
    df["target_id"] = df.apply(
        lambda x: x[f"object_{x.target_object_index}_id"],
        axis=1,
    )

    def _get_response_id(row):
        if np.isnan(row.response_location):
            return np.nan
        for i in range(3):
            if row[f"object_{i}_location"] == row.response_location:
                return row[f"object_{i}_id"]

    df["response_id"] = df.apply(_get_response_id, axis=1)

    # Add reaction time
    df["reaction_time"] = df["response_time"] - df["time_cue_onset"]

    return df
