"""Classes for filtering which trials to use for modeling."""

import numpy as np


class TriangleTrialFilter:
    """Filter for Triangle dataset."""

    def __call__(self, df_behavior):
        """Filter trials.

        Args:
            df_behavior: DataFrame containing behavior data.

        Returns:
            Filtered DataFrame with only trials that are completed and on the
            triangle task.
        """
        return df_behavior[(df_behavior.completed) & (df_behavior.on_triangle)]


class RingTrialFilter:
    """Filter for Ring dataset."""

    def __init__(self, avoid_lr: bool = True):
        """Constructor.

        Args:
            avoid_lr: If True, avoid left/right trials which are
                over-represented in the data.
        """
        self.avoid_lr = avoid_lr

    def __call__(self, df_behavior):
        """Filter trials.

        Args:
            df_behavior: DataFrame containing behavior data.

        Returns:
            Filtered DataFrame with only trials that are completed and (if
            necessary) not left/right trials.
        """
        df_behavior = df_behavior[df_behavior.completed]

        if self.avoid_lr:
            # Reject if this is a 2-object left/right trial, because those are
            # over-represented in the data
            reject_indices = df_behavior[
                (df_behavior.num_objects == 2)
                & np.isclose(df_behavior.object_0_theta, 0)
                & np.isclose(df_behavior.object_1_theta, np.pi)
            ]
            df_behavior = df_behavior.drop(reject_indices.index)
            reject_indices = df_behavior[
                (df_behavior.num_objects == 2)
                & np.isclose(df_behavior.object_1_theta, 0)
                & np.isclose(df_behavior.object_0_theta, np.pi)
            ]
            df_behavior = df_behavior.drop(reject_indices.index)

        return df_behavior
