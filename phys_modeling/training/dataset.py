"""Dataset class."""

import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
import constants


class Dataset:
    IDENTITY_ONEHOT = {
        "a": [1, 0, 0],
        "b": [0, 1, 0],
        "c": [0, 0, 1],
        np.nan: [0, 0, 0],
    }

    def __init__(
        self,
        subject: str,
        session: str,
        phase: str,
        unit_filter: callable,
        trial_filter: callable,
        max_n_objects: int,
        min_trials_per_unit: int = 200,
        min_units_per_trial: int = 5,
        test_fraction: float = 0.2,
        random_seed: int = 0,
    ):
        """Constructor.

        Args:
            subject: Subject name.
            session: Session date.
            phase: Phase of the experiment (e.g., "delay").
            unit_filter: Function to filter units.
            trial_filter: Function to filter trials.
            max_n_objects: Maximum number of objects per trial.
            min_trials_per_unit: Minimum number of trials per unit.
            min_units_per_trial: Minimum number of units per trial.
            test_fraction: Fraction of trials to use for testing.
            random_seed: Random seed for reproducibility.
        """
        # Set random seed
        np.random.seed(random_seed)

        # Register variables
        self._subject = subject
        self._session = session
        self._phase = phase
        self._unit_filter = unit_filter
        self._trial_filter = trial_filter
        self._max_n_objects = max_n_objects
        self._min_trials_per_unit = min_trials_per_unit
        self._min_units_per_trial = min_units_per_trial
        self._test_fraction = test_fraction
        self._random_seed = random_seed

        # Load behavior data
        df_behavior_triangle = pd.read_csv(constants.BEHAVIOR_DIR_TRIANGLE)
        df_behavior_ring = pd.read_csv(constants.BEHAVIOR_DIR_RING)
        df_behavior_triangle = df_behavior_triangle[
            (df_behavior_triangle.subject == self._subject)
            & (df_behavior_triangle.session == self._session)
        ]
        df_behavior_ring = df_behavior_ring[
            (df_behavior_ring.subject == self._subject)
            & (df_behavior_ring.session == self._session)
        ]
        if len(df_behavior_triangle) > 0:
            print("Triangle session")
            df_behavior = df_behavior_triangle.reset_index()
        elif len(df_behavior_ring) > 0:
            print("Ring session")
            df_behavior = df_behavior_ring.reset_index()
        else:
            raise ValueError(
                f"No behavior data found for subject {self._subject} and "
                f"session {self._session}"
            )

        # Filter trials
        df_behavior = self._trial_filter(df_behavior).reset_index()
        trials = df_behavior.trial_num.values
        trial_to_index = {t: i for i, t in enumerate(trials)}
        print(f"Original number of trials: {len(df_behavior)}")

        # Load units data
        selectivity_path = (
            constants.SELECTIVITY_DIR
            / self._subject
            / self._session
            / "units.csv"
        )
        df_units = pd.read_csv(selectivity_path, index_col=0)
        print(f"Original number of units: {len(df_units)}")

        # Filter units
        df_units = self._unit_filter(df_units).reset_index()
        units = df_units.unit.values

        # Load neural data
        spike_counts_per_unit, trials_per_unit = self._get_spike_count_data()

        # Create neural data matrix
        spike_count_timeframe_ms = constants.SPIKE_COUNT_TIMEFRAME_PER_PHASE[
            self._phase
        ]
        spike_count_start_bin = (
            spike_count_timeframe_ms[0] // constants.SPIKE_COUNT_BIN_SIZE_MS
        )
        spike_count_end_bin = (
            spike_count_timeframe_ms[1] // constants.SPIKE_COUNT_BIN_SIZE_MS
        )
        num_spike_count_bins = spike_count_end_bin - spike_count_start_bin
        neural = np.nan * np.ones(
            (len(trials), len(units), num_spike_count_bins)
        )
        for unit_index, unit in enumerate(units):
            unit_spike_counts = spike_counts_per_unit[unit]
            for i, trial_num in enumerate(trials_per_unit[unit]):
                if trial_num not in trial_to_index:
                    continue
                trial_index = trial_to_index[trial_num]
                trial_spike_counts = unit_spike_counts[i]
                neural[trial_index, unit_index, :] = trial_spike_counts[
                    spike_count_start_bin:spike_count_end_bin
                ]

        # Filter trials and units
        presence = ~np.isnan(neural[:, :, 0])
        units_to_remove, trials_to_remove = self._filter_trials_and_units(
            presence
        )
        neural = np.delete(neural, units_to_remove, axis=1)
        neural = np.delete(neural, trials_to_remove, axis=0)
        units = np.delete(units, units_to_remove)
        trials = np.delete(trials, trials_to_remove)
        df_behavior = df_behavior.drop(np.argwhere(trials_to_remove).flatten())
        df_units = df_units.drop(np.argwhere(units_to_remove).flatten())
        print(f"Number of trials after filtering: {len(trials)}")
        print(f"Number of units after filtering: {len(units)}")

        # Make sure we have enough units and trials
        if len(units) == 0:
            raise ValueError("No units left after filtering")
        if len(trials) == 0:
            raise ValueError("No trials left after filtering")

        # Get stimuli
        self._n_trials, self._n_units, self._n_timesteps = neural.shape
        self._data = self._get_data_dict(df_behavior, neural)

        # Make train/test split
        self._train_data, self._test_data, self._train_mask = self._train_test(
            self._data
        )

        # Convert data to torch tensors
        self._train_data_torch = {
            key: torch.from_numpy(value.astype(np.float32))
            for key, value in self._train_data.items()
        }
        self._test_data_torch = {
            key: torch.from_numpy(value.astype(np.float32))
            for key, value in self._test_data.items()
        }
        self._train_data_torch["trial"] = self._train_data_torch["trial"].type(
            torch.int64
        )
        self._test_data_torch["trial"] = self._test_data_torch["trial"].type(
            torch.int64
        )
        self._train_data_torch["time_indices"] = self._train_data_torch[
            "time_indices"
        ].type(torch.int64)
        self._test_data_torch["time_indices"] = self._test_data_torch[
            "time_indices"
        ].type(torch.int64)

        # Register variables
        self._df_behavior = df_behavior
        self._df_units = df_units
        self._valid_objects = np.isfinite(self._data["positions"][:, 0, :, 0])
        self._valid_objects_torch = torch.from_numpy(
            self._valid_objects.astype(np.float32)
        )

    def _get_spike_count_data(self):
        """Load spike counts for the session."""
        spike_counts_per_unit = {}
        trials_per_unit = {}
        spike_count_session_dir = (
            constants.SPIKES_PER_TRIAL_DIR / self._subject / self._session
        )
        for probe_dir in (spike_count_session_dir).iterdir():
            if probe_dir.name.startswith("."):
                continue
            for quality_dir in probe_dir.iterdir():
                if quality_dir.name.startswith("."):
                    continue
                for spike_counts_file in quality_dir.iterdir():
                    if not "spike_counts" in spike_counts_file.name:
                        continue
                    trials_filename = spike_counts_file.name.replace(
                        "spike_counts", "trials"
                    )
                    trials_file = quality_dir / f"{trials_filename}"
                    unit = int(trials_filename.split("_")[0])
                    spike_counts_per_unit[unit] = pickle.load(
                        open(spike_counts_file, "rb")
                    )
                    trials_per_unit[unit] = pickle.load(
                        open(trials_file, "rb")
                    )
        return spike_counts_per_unit, trials_per_unit

    def _get_data_dict(self, df_behavior, neural):
        """Process trials to get stimuli."""
        # [n_trials, n_timesteps, n_units]
        neural = np.transpose(neural, (0, 2, 1))
        neural_finite = ~np.isnan(neural)
        neural[np.isnan(neural)] = 0.0

        # Create time, position, and identity arrays
        time_indices = np.arange(self._n_timesteps)
        time = constants.SPIKE_COUNT_BIN_SIZE_MS / 1000 * time_indices
        time = np.tile(time, (self._n_trials, 1))
        time_indices = np.tile(time_indices, (self._n_trials, 1))
        trial = np.arange(self._n_trials)
        trial = np.tile(trial[:, None], (1, self._n_timesteps))
        positions = np.stack(
            [
                [
                    df_behavior[f"object_{i}_x"].values,
                    df_behavior[f"object_{i}_y"].values,
                ]
                for i in range(self._max_n_objects)
            ],
            axis=0,
        )
        # [n_trials, n_objects, 2]
        positions = np.transpose(positions, (2, 0, 1))
        positions = np.tile(positions[:, None], (1, self._n_timesteps, 1, 1))
        identities = np.stack(
            [
                df_behavior[f"object_{i}_id"].values
                for i in range(self._max_n_objects)
            ],
            axis=1,
        )
        identities = np.array(
            [[self.IDENTITY_ONEHOT[id_] for id_ in row] for row in identities]
        )
        identities = np.tile(identities[:, None], (1, self._n_timesteps, 1, 1))

        # Create data dictionary
        data_dict = {
            "time": time,  # [n_trials, n_timesteps]
            "time_indices": time_indices,  # [n_trials, n_timesteps]
            "positions": positions,  # [n_trials, timesteps, n_objects, 2]
            "identities": identities,  # [n_trials, timesteps, n_objects, 3]
            "trial": trial,  # [n_trials, n_timesteps]
            "neural": neural,  # [n_trials, n_timesteps, n_units]
            "neural_finite": neural_finite,  # [n_trials, n_timesteps, n_units]
        }

        return data_dict

    def _train_test(self, data_dict):
        """Split data into train and test sets."""
        # Sample a train/test mask
        train_mask = (
            np.random.rand(self._n_trials, self._n_units) > self._test_fraction
        )

        # Correct train_mask to have at least 3 units per trial
        for i in range(self._n_trials):
            mask_row = train_mask[i, :]
            finite_row = data_dict["neural_finite"][i, 0]
            need_to_convert = 3 - (mask_row * finite_row).sum()
            if need_to_convert > 0:
                test_units = np.argwhere(~mask_row * finite_row).flatten()
                sample = np.random.choice(
                    test_units, size=need_to_convert, replace=False
                )
                train_mask[i, sample] = True

        # Tile mask
        train_mask = np.repeat(train_mask[:, None], self._n_timesteps, axis=1)
        train_data = {key: np.copy(value) for key, value in data_dict.items()}
        test_data = {key: np.copy(value) for key, value in data_dict.items()}
        train_data["neural"][~train_mask] = 0.0
        train_data["neural_finite"][~train_mask] = False
        test_data["neural"][train_mask] = 0.0
        test_data["neural_finite"][train_mask] = False

        # Remove NaN values from positions and identities
        train_data["positions"][np.isnan(train_data["positions"])] = 0.0
        test_data["positions"][np.isnan(test_data["positions"])] = 0.0

        return train_data, test_data, train_mask[:, 0]

    def _filter_trials_and_units(self, presence):
        """Get trials and units to remove.

        Args:
            presence: Boolean numpy array of shape (n_trials, n_units)
                indicating whether each unit was present in each trial.

        Returns:
            units_to_remove: Numpy array of shape (n_units,) indicating
                whether each unit should be removed.
            trials_to_remove: Numpy array of shape (n_trials,) indicating
                whether each trial should be
        """
        trials_per_unit = np.sum(presence, axis=0)
        units_per_trial = np.sum(presence, axis=1)
        units_to_remove = trials_per_unit < self._min_trials_per_unit
        trials_to_remove = units_per_trial < self._min_units_per_trial
        delta = np.sum(presence[:, units_to_remove]) + np.sum(
            presence[trials_to_remove, :]
        )
        if delta == 0:
            return units_to_remove, trials_to_remove
        else:
            presence[:, units_to_remove] = False
            presence[trials_to_remove, :] = False
            return self._filter_trials_and_units(presence)

    def get_batch(self, train: bool, batch_size: int = None):
        """Get a batch of training data.

        Args:
            train: Boolean indicating whether to get training data (True) or
                testing data (False).
            batch_size: Number of trials to return in the batch. If None, return
                all trials.

        Returns:
            A tuple containing:
                - time: Numpy array of shape [batch_size, n_objects] with time
                  values.
                - time_indices: Numpy array of shape [batch_size, n_objects]
                  with time indices.
                - positions: Numpy array of shape [batch_size, n_objects, 2]
                  with position values.
                - identities: Numpy array of shape [batch_size, n_objects, 3]
                  with identity values.
                - trial: Numpy array of shape [batch_size] with trial indices.
                - neural: Numpy array of shape [batch_size, n_units]
                  with neural data.
                - neural_finite: Numpy array of shape [batch_size, n_units]
                  indicating whether the neural data is finite.
            If batch_size is None, all of these return arrays have shape
            [n_trials, n_timesteps, ...].
        """
        if train:
            data = self._train_data_torch
        else:
            data = self._test_data_torch

        if batch_size is not None:
            indices_trials = np.random.choice(
                self._n_trials, size=batch_size, replace=True
            )
            indices_timesteps = np.random.choice(
                self._n_timesteps, size=batch_size, replace=True
            )
            indices = np.array([indices_trials, indices_timesteps]).T
            data = {
                key: value[indices[:, 0], indices[:, 1]]
                for key, value in data.items()
            }

        return data

    def cache(self, write_dir: str):
        """Write positions, identities, and train mask."""
        write_dir.mkdir(parents=True, exist_ok=True)
        positions = self._data["positions"][:, 0]
        identities = self._data["identities"][:, 0]
        train_mask = self._train_mask
        trial_num = self.df_behavior.trial_num.values
        np.save(write_dir / "positions.npy", positions)
        np.save(write_dir / "identities.npy", identities)
        np.save(write_dir / "train_mask.npy", train_mask)
        np.save(write_dir / "trial_num.npy", trial_num)
        return write_dir

    @property
    def unit_filter(self):
        return self._unit_filter

    @property
    def trial_filter(self):
        return self._trial_filter

    @property
    def data(self):
        """Return the data."""
        return self._data

    @property
    def train_data(self):
        """Return the training data."""
        return self._train_data

    @property
    def test_data(self):
        """Return the testing data."""
        return self._test_data

    @property
    def train_mask(self):
        """Return the training mask."""
        return self._train_mask

    @property
    def df_behavior(self):
        """Return the behavior data."""
        return self._df_behavior

    @property
    def df_units(self):
        """Return the units data."""
        return self._df_units

    @property
    def n_trials(self):
        """Return the number of trials."""
        return self._n_trials

    @property
    def n_timesteps(self):
        """Return the number of timesteps."""
        return self._n_timesteps

    @property
    def n_units(self):
        """Return the number of units."""
        return self._n_units

    @property
    def max_n_objects(self):
        """Return the maximum number of objects per trial."""
        return self._max_n_objects

    @property
    def sample_rate_ms(self):
        return constants.SPIKE_COUNT_BIN_SIZE_MS

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def subject(self):
        return self._subject

    @property
    def session(self):
        return self._session

    @property
    def phase(self):
        return self._phase

    @property
    def valid_objects(self):
        return self._valid_objects

    @property
    def valid_objects_torch(self):
        return self._valid_objects_torch
