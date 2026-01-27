"""Generator for synthetic datasets."""

import json
import logging
import sys

import numpy as np
import torch

sys.path.append("..")
sys.path.append("../training")
from config_utils import build_from_config
from models import abstract_model


class Generator:
    """Callable for generating synthetic datasets."""

    def __init__(self, log_dir: str, random_seed: int = 0):
        """Constructor.

        Args:
            log_dir: Directory to save generated data.
            random_seed: Random seed for reproducibility.
        """
        self._log_dir = log_dir
        self._random_seed = random_seed
        np.random.seed(random_seed)

        # Create model from logged config
        config = json.load(open(f"{log_dir}/config.json"))
        model_config = config["kwargs"]["model"]
        model = build_from_config.build_from_config(model_config)

        # Load model from snapshot
        snapshot_path = f"{log_dir}/stop_step/model_snapshot.pth"
        model.load_state_dict(torch.load(snapshot_path))

        # Register model
        self._model = model

    def __call__(self, log_dir: str):
        """Generate synthetic dataset.

        Args:
            log_dir: Directory to save generated data.
        """
        logging.info("\nBeginning dataset generation")

        # Get continuous firing rate prediction
        pred_train, _, neural_finite_train = (
            self._model.prediction_per_unit_numpy(train=True)
        )
        pred_test, _, neural_finite_test = (
            self._model.prediction_per_unit_numpy(train=False)
        )
        neural_finite_train = neural_finite_train.astype(bool)
        neural_finite_test = neural_finite_test.astype(bool)

        # Combine train and test
        pred = np.copy(pred_train)
        pred[neural_finite_test] = pred_test[neural_finite_test]
        neural_finite = neural_finite_train + neural_finite_test
        pred[~neural_finite] = 0.0

        # Sample Poisson spikes from prediction
        spikes = np.random.poisson(pred).astype(np.float32)
        pred[~neural_finite] = np.nan
        spikes[~neural_finite] = np.nan

        # Save spikes to log_dir
        logging.info(f"\nSaving spikes to {log_dir}")
        np.save(f"{log_dir}/spikes.npy", spikes)

        # Save firing rates to log_dir
        logging.info(f"\nSaving firing rates to {log_dir}")
        np.save(f"{log_dir}/firing_rates.npy", pred)

        # Evaluate log likelihood per unit trial
        log_likelihood = abstract_model.poisson_log_likelihood(spikes, pred)
        mll = np.sum(log_likelihood, axis=2)
        logging.info(f"\nSaving log likelihood to {log_dir}")
        np.save(f"{log_dir}/log_likelihood.npy", mll)

        logging.info(f"\nFinished evaluating ground truth log likelihood.")
