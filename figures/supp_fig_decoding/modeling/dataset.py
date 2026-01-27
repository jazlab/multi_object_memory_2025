"""Dataset class."""

import enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_BEHAVIOR_DIR_TRIANGLE = Path(
    "../../../cache/behavior_processing/triangle.csv"
)
_BEHAVIOR_DIR_RING = Path("../../../cache/behavior_processing/ring.csv")
_SELECTIVITY_DIR = Path("../../../cache/phys_processing/selectivity")
_NEURAL_DATA_DIR = Path("../../../cache/phys_processing/spikes_to_trials")


class Mode(enum.Enum):
    NEURAL = "neural"
    RECEPTIVE_FIELDS = "receptive_fields"
    ORTHOGONAL = "orthogonal"


def get_simultaneous(presence, keep_units_weight=1.0):
    """Get units that are recorded simultaneously.

    Args:
        presence: Array of shape (n_trials, n_units) where 1 indicates presence.
        keep_units_weight: Weight to keep units over trials.
    """
    n_trials, n_units = presence.shape
    unit_indices = list(range(n_units))
    trial_indices = list(range(n_trials))
    discard_units = []
    discard_trials = []

    done = False
    while not done:
        mean_presence_per_unit = presence.mean(axis=0)
        mean_presence_per_trial = presence.mean(axis=1)
        worst_unit = np.argmin(mean_presence_per_unit)
        worst_trial = np.argmin(mean_presence_per_trial)
        worst_unit_presence = mean_presence_per_unit[worst_unit]
        worst_trial_presence = mean_presence_per_trial[worst_trial]

        if worst_unit_presence == 1 and worst_trial_presence == 1:
            done = True
            break

        if worst_trial_presence < keep_units_weight * worst_unit_presence:
            discard_trial_index = trial_indices[worst_trial]
            discard_trials.append(discard_trial_index)
            presence = np.delete(presence, worst_trial, axis=0)
            trial_indices.pop(worst_trial)
        else:
            discard_unit_index = unit_indices[worst_unit]
            discard_units.append(discard_unit_index)
            presence = np.delete(presence, worst_unit, axis=1)
            unit_indices.pop(worst_unit)

    return discard_units, discard_trials


class Dataset:
    IDENTITY_ONEHOT = {
        "a": [1, 0, 0],
        "b": [0, 1, 0],
        "c": [0, 0, 1],
    }

    def __init__(
        self,
        subject: str,
        session: str,
        phase: str,
        num_objects: int,
        labeler: callable,
        min_trials_per_unit: int = 50,
        min_units_per_trial: int = 5,
        test_fraction: float = 0.2,
        shuffle_labels: bool = False,
        random_seed: int = 0,
        mode: Mode = Mode.NEURAL.value,
        brain_area=None,
    ):
        """Constructor."""
        # Set random seed
        np.random.seed(random_seed)

        # Register variables
        self._subject = subject
        self._session = session
        self._phase = phase
        self._num_objects = num_objects
        self._labeler = labeler
        self._min_trials_per_unit = min_trials_per_unit
        self._min_units_per_trial = min_units_per_trial
        self._test_fraction = test_fraction
        self._shuffle_labels = shuffle_labels
        self._random_seed = random_seed
        self._mode = mode
        self._brain_area = brain_area

        # Compute neural data path
        self._neural_data_path = (
            _NEURAL_DATA_DIR
            / f"{self._phase}_phase_firing_rates"
            / self._subject
            / self._session
            / "fr.csv"
        )

        # Load behavior data
        df_behavior_triangle = pd.read_csv(_BEHAVIOR_DIR_TRIANGLE)
        df_behavior_triangle = df_behavior_triangle[
            (df_behavior_triangle.subject == self._subject)
            & (df_behavior_triangle.session == self._session)
        ]
        df_behavior_ring = pd.read_csv(_BEHAVIOR_DIR_RING)
        df_behavior_ring = df_behavior_ring[
            (df_behavior_ring.subject == self._subject)
            & (df_behavior_ring.session == self._session)
        ]
        if len(df_behavior_triangle) > 0:
            df_behavior = df_behavior_triangle
            df_behavior = df_behavior[df_behavior.on_triangle]
            self._task = "triangle"
        elif len(df_behavior_ring) > 0:
            df_behavior = df_behavior_ring
            self._task = "ring"
        else:
            raise ValueError(
                f"No behavior data found for subject {self._subject} and "
                f"session {self._session}"
            )

        # Filter trials
        df_behavior = df_behavior[df_behavior.completed]

        # Get labels
        labels = df_behavior.apply(self._row_to_label, axis=1)
        keep_indices = np.array([x is not None for x in labels])
        df_behavior = df_behavior[keep_indices]
        labels = np.stack(labels[keep_indices])
        trials = df_behavior.trial_num.values
        print(f"Original number of trials: {len(trials)}")

        # Filter units
        selectivity_path = (
            _SELECTIVITY_DIR / self._subject / self._session / "units.csv"
        )
        df_units = pd.read_csv(selectivity_path, index_col=0)
        print(f"Original number of units: {len(df_units)}")
        df_units = df_units[df_units.significant]
        if self._brain_area == "DMFC":
            df_units = df_units[df_units["probe"].str.contains("s")]
        elif self._brain_area == "FEF":
            df_units = df_units[df_units["probe"].str.contains("vprobe")]
        elif self._brain_area is not None:
            raise ValueError(f"Unknown brain area: {self._brain_area}")

        # Load neural data of shape (n_trials, n_units)
        units = df_units.unit.values
        neural_data = self._get_neural_data(trials, units)

        # Filter units by number of trials
        n_trials_per_unit = np.sum(~np.isnan(neural_data), axis=0)
        keep_units = n_trials_per_unit >= self._min_trials_per_unit
        units = units[keep_units]
        neural_data = neural_data[:, keep_units]
        print(f"Number of units after min_trials_per_unit: {len(units)}")

        # Filter trials by number of units
        n_units_per_trial = np.sum(~np.isnan(neural_data), axis=1)
        keep_trials = n_units_per_trial >= self._min_units_per_trial
        trials = trials[keep_trials]
        labels = labels[keep_trials]
        neural_data = neural_data[keep_trials, :]
        print(f"Number of trials after min_units_per_trial: {len(trials)}")

        # Keep only simultaneously recorded units
        discard_units, discard_trials = get_simultaneous(
            np.isfinite(neural_data),
            keep_units_weight=1.0,
        )
        units = np.delete(units, discard_units)
        trials = np.delete(trials, discard_trials)
        labels = np.delete(labels, discard_trials, axis=0)
        neural_data = np.delete(neural_data, discard_trials, axis=0)
        neural_data = np.delete(neural_data, discard_units, axis=1)
        print(f"Number of units after simultaneous filtering: {len(units)}")
        print(f"Number of trials after simultaneous filtering: {len(trials)}")

        # If needed randomize labels
        if self._shuffle_labels:
            shuffle_indices = np.arange(len(trials))
            np.random.shuffle(shuffle_indices)
            labels = labels[shuffle_indices]

        # Normalize neural data
        # mean_firing_rates = np.nanmean(neural_data, axis=0)
        # std_firing_rates = np.nanstd(neural_data, axis=0)
        # neural_data = (neural_data - mean_firing_rates) / std_firing_rates

        # Override neural data with hand-crafted datasets if model is not
        # "neural"
        mean_fr_per_unit = np.mean(neural_data, axis=0)
        var_fr_per_unit = np.var(neural_data, axis=0)
        n_trials, n_units = neural_data.shape[:2]
        if self._mode == Mode.RECEPTIVE_FIELDS.value:
            # Override neural data with receptive fields
            if labels.ndim != 3:
                raise ValueError(
                    "Orthogonal mode only implemented for 2-object tasks."
                )
            if self._task == "triangle":
                labels_sum = labels.sum(axis=1)
                receptive_fields = np.exp(
                    np.random.normal(loc=0, scale=1, size=(3, n_units))
                )
                neural_data = np.dot(labels_sum, receptive_fields)
            else:
                rf_centers = np.random.uniform(
                    low=-1, high=1, size=(2, n_units)
                )
                proximities = np.zeros((n_trials, n_units, 2))
                for i_trial in range(n_trials):
                    pos_0, pos_1 = labels[i_trial]
                    proximities[i_trial, :, 0] = np.linalg.norm(
                        rf_centers - pos_0[:, np.newaxis], axis=0
                    )
                    proximities[i_trial, :, 1] = np.linalg.norm(
                        rf_centers - pos_1[:, np.newaxis], axis=0
                    )
                neural_data = np.sum(np.exp(-proximities), axis=2)
        elif self._mode == Mode.ORTHOGONAL.value:
            # Override neural data with orthogonal data
            if labels.ndim != 3:
                raise ValueError(
                    "Orthogonal mode only implemented for 2-object tasks."
                )
            half_n_units = n_units // 2
            if self._task == "triangle":
                receptive_fields = np.exp(
                    np.random.normal(loc=0, scale=1, size=(3, n_units))
                )
                neural_data = np.stack(
                    [
                        np.dot(labels[:, 0], receptive_fields),
                        np.dot(labels[:, 1], receptive_fields),
                    ],
                    axis=1,
                )
                for i in range(neural_data.shape[0]):
                    if np.random.rand() < 0.5:
                        neural_data[i, 1, :half_n_units] = 0
                        neural_data[i, 0, half_n_units:] = 0
                    else:
                        neural_data[i, 0, :half_n_units] = 0
                        neural_data[i, 1, half_n_units:] = 0
                neural_data = neural_data.sum(axis=1)
            else:
                rf_centers = np.random.uniform(
                    low=-1, high=1, size=(2, n_units)
                )
                proximities = np.zeros((n_trials, n_units, 2))
                for i_trial in range(n_trials):
                    pos_0, pos_1 = labels[i_trial]
                    proximities[i_trial, :, 0] = np.linalg.norm(
                        rf_centers - pos_0[:, np.newaxis], axis=0
                    )
                    proximities[i_trial, :, 1] = np.linalg.norm(
                        rf_centers - pos_1[:, np.newaxis], axis=0
                    )
                weights = np.exp(-proximities)
                scales = np.random.uniform(low=1, high=30, size=(n_units))
                neural_data = weights * scales[np.newaxis, :, np.newaxis]
                for i_trial in range(n_trials):
                    if np.random.rand() < 0.5:
                        neural_data[i_trial, :half_n_units, 0] = 0
                        neural_data[i_trial, half_n_units:, 1] = 0
                    else:
                        neural_data[i_trial, :half_n_units, 1] = 0
                        neural_data[i_trial, half_n_units:, 0] = 0
                neural_data = neural_data.sum(axis=2)
        elif self._mode != Mode.NEURAL.value:
            raise ValueError(f"Unknown mode: {self._mode}")

        if self._mode != Mode.NEURAL.value:
            # Normalize synthetic neural data to match original mean and var
            neural_data_mean = np.mean(neural_data, axis=0)
            neural_data_var = np.var(neural_data, axis=0)
            neural_data = (
                neural_data - neural_data_mean[np.newaxis, :]
            ) / np.sqrt(neural_data_var[np.newaxis, :] + 1e-8) * np.sqrt(
                var_fr_per_unit[np.newaxis, :]
            ) + mean_fr_per_unit[
                np.newaxis, :
            ]

        # Split train and test trials
        n_trials = len(trials)
        shuffled_indices = np.arange(n_trials)
        np.random.shuffle(shuffled_indices)
        n_test = int(self._test_fraction * n_trials)
        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]
        self._train_labels_np = labels[train_indices]
        self._test_labels_np = labels[test_indices]
        self._train_neural_np = neural_data[train_indices, :]
        self._test_neural_np = neural_data[test_indices, :]

        # Convert to torch tensors
        self._train_labels = torch.tensor(
            self._train_labels_np,
            dtype=torch.float32,
        )
        self._test_labels = torch.tensor(
            self._test_labels_np,
            dtype=torch.float32,
        )
        self._train_neural = torch.tensor(
            self._train_neural_np,
            dtype=torch.float32,
        )
        self._test_neural = torch.tensor(
            self._test_neural_np,
            dtype=torch.float32,
        )

        # Register dimensions
        self._n_units = self._train_neural.shape[1]
        self._label_dim = self._train_labels.shape[-1]

    def _row_to_label(self, row):
        """Convert a behavior dataframe row to a label."""
        return self._labeler(self, row)

    def _get_neural_data(self, trials, units):
        """Load neural data for the session."""
        neural_data = pd.read_csv(self._neural_data_path)
        neural_data = neural_data[
            neural_data.trial.isin(trials) & neural_data.unit.isin(units)
        ]
        n_trials = len(trials)
        n_units = len(units)
        neural = np.nan * np.ones((n_trials, n_units))
        for i_unit, unit in enumerate(units):
            df_unit = neural_data[neural_data.unit == unit]
            for i_trial, trial in enumerate(trials):
                df_trial = df_unit[df_unit.trial == trial]
                if len(df_trial) == 0:
                    continue
                firing_rate = df_trial.firing_rate.values[0]
                neural[i_trial, i_unit] = firing_rate

        return neural

    def get_batch(self, batch_size, train: bool = True):
        """Get a batch of data from the dataset."""
        # Select train or test data
        if train:
            neural = self._train_neural
            labels = self._train_labels
        else:
            neural = self._test_neural
            labels = self._test_labels

        # Sample batch
        if batch_size is None:
            batch_neural = neural
            batch_labels = labels
            batch_indices = None
        else:
            n_samples = neural.shape[0]
            batch_indices = np.random.choice(
                n_samples,
                size=batch_size,
                replace=True,
            )
            batch_neural = neural[batch_indices, :]
            batch_labels = labels[batch_indices, :]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)

        # Return batch
        return {
            "neural": batch_neural,
            "labels": batch_labels,
            "indices": batch_indices,
        }

    @property
    def n_units(self):
        """Get number of units."""
        return self._n_units

    @property
    def label_dim(self):
        """Get label dimension."""
        return self._label_dim

    @property
    def num_objects(self):
        """Get number of objects."""
        return self._num_objects

    @property
    def random_seed(self):
        """Get random seed."""
        return self._random_seed

    @property
    def n_train_trials(self):
        """Get number of training trials."""
        return self._train_neural.shape[0]

    @property
    def task(self):
        """Get task type."""
        return self._task
