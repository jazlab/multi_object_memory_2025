"""Callables for converting dataframe trials to stimulus vectors."""

import sys

import numpy as np

sys.path.append("..")
import constants


class TriangleObjectConditions:
    """Class for Triangle dataset converting trials to stimulus vectors.

    Each trial's stimulus is converted into a vector of integers, one for each
    object, where each integer is an index of the condition of the object.
    """

    def __init__(self, condition_on_identity: bool = True):
        """Constructor.

        Args:
            condition_on_identity: If True, condition on the identity of the
                object.
        """
        self._condition_on_identity = condition_on_identity
        self._identity_to_index = {
            k: i for i, k in enumerate(constants.IDENTITIES)
        }
        self._conditions = []
        for location in range(3):
            if self._condition_on_identity:
                for identity in constants.IDENTITIES:
                    self._conditions.append((location, identity))
            else:
                self._conditions.append(location)
        self._condition_to_index = {
            c: i for i, c in enumerate(self._conditions)
        }

    def __call__(self, df_behavior):
        """Convert trials to stimulus vectors.

        Args:
            df_behavior: DataFrame containing behavior data.

        Returns:
            A numpy array of shape (n_trials, n_objects) containing the
                condition indices for each object in each trial. For trials with
                fewer than 3 objects, the remaining indices will be NaN.
        """
        condition_indices = []

        for _, row in df_behavior.iterrows():
            condition_per_object = np.nan * np.ones(3)
            num_objects = row.num_objects
            for i in range(num_objects):
                location = int(row[f"object_{i}_location"])
                if self._condition_on_identity:
                    identity = row[f"object_{i}_id"]
                    condition_per_object[i] = self._condition_to_index[
                        (location, identity)
                    ]
                else:
                    condition_per_object[i] = self._condition_to_index[
                        location
                    ]
            condition_indices.append(condition_per_object)

        condition_indices = np.array(condition_indices)

        return condition_indices


class RingObjectCentric:
    """Class for Ring dataset converting trials to stimulus vectors.

    Each trial's stimulus is converted into an array of features, one for each
    object, where each feature is a vector of [theta, identity] if
    condition_on_identity is True, or [theta, 0] if it is False.
    """

    def __init__(self, condition_on_identity: bool = True):
        """Constructor.

        Args:
            condition_on_identity: If True, condition on the identity of the
                object.
        """
        self._condition_on_identity = condition_on_identity
        self._identity_to_index = {
            k: i for i, k in enumerate(constants.IDENTITIES)
        }

    def _get_features(self, theta: float, identity: str) -> np.ndarray:
        """Get features for a single object.

        Args:
            theta: The angle of the object in radians.
            identity: The identity of the object.

        Returns:
            A numpy array of shape (2,) containing the features for the object.
        """
        if self._condition_on_identity:
            identity = self._identity_to_index[identity]
            features = np.array([theta, identity])
        else:
            features = np.array([theta, 0])
        return features

    def __call__(self, df_behavior) -> np.ndarray:
        """Convert trials to stimulus vectors.

        Args:
            df_behavior: DataFrame containing behavior data.

        Returns:
            A numpy array of shape (n_trials, max_num_objects,
            num_features_per_object) containing the features for each object in
            each trial. For trials with fewer than max_num_objects, the
            remaining features will be NaN.
        """
        object_features = []

        for _, row in df_behavior.iterrows():
            features = np.nan * np.ones(
                (self.max_num_objects, self.num_features_per_object)
            )
            num_objects = row.num_objects
            for i in range(num_objects):
                theta = row[f"object_{i}_theta"]
                identity = row[f"object_{i}_id"]
                features[i] = self._get_features(theta, identity)
            object_features.append(features)

        object_features = np.array(object_features)

        return object_features

    @property
    def num_features_per_object(self) -> int:
        """Number of features per object."""
        return 2

    @property
    def num_identities(self) -> int:
        """Number of identities."""
        if self._condition_on_identity:
            return 3
        else:
            return 1

    @property
    def max_num_objects(self) -> int:
        """Maximum number of objects in a trial."""
        return 2
