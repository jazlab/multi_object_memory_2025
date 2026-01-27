"""Abstract model class.

All models should inherit from the AbstractModel class and implement the methods
defined there.
"""

import abc

import numpy as np
import torch
from scipy.special import factorial

from . import embedding_mlp


def poisson_log_likelihood(
    target: np.ndarray,
    pred: np.ndarray,
    epsilon: float = 1e-6,
):
    """Compute the Poisson log likelihood.

    Args:
        target: Numpy array of target values (e.g., spike counts).
        pred: Numpy array of predicted values (e.g., firing rates).
        epsilon: Small value to avoid log(0).

    Returns:
        Numpy array of the log likelihood values.
    """
    if target.shape != pred.shape:
        raise ValueError("Target and prediction shapes must match.")
    target = np.copy(target)
    target[target < 0] = 0.0
    pred[pred < epsilon] = epsilon
    log_factorial_target = np.log(factorial(target.astype(int)))
    log_likelihood = target * np.log(pred) - pred - log_factorial_target
    return log_likelihood


class AbstractModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for neural data modeling."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        time_noise_sigma: float = 0.2,
        position_noise_sigma: float = 0.2,
        epsilon: float = 1e-6,
        softplus_beta: float = 5,
        **embedding_mlp_kwargs
    ):
        """Constructor.

        Args:
            dataset: Dataset object containing neural data and stimuli. Must be
                an instance of phys_modeling.training.dataset.Dataset.
            time_noise_sigma: Standard deviation of noise for time input.
            position_noise_sigma: Standard deviation of noise for space input.
            epsilon: Small value to avoid log(0) in likelihood calculations.
            softplus_beta: Beta parameter for the softplus nonlinearity.
        """
        # Set random seed
        np.random.seed(dataset.random_seed)
        torch.manual_seed(dataset.random_seed)

        # Initialize the parent class and register variables
        super(AbstractModel, self).__init__()
        self._dataset = dataset
        self._batch_size = batch_size
        self._time_noise_sigma = time_noise_sigma
        self._position_noise_sigma = position_noise_sigma
        self._epsilon = epsilon
        self._softplus_beta = softplus_beta

        # Nonlinearity
        softplus = torch.nn.Softplus(beta=self._softplus_beta)
        self._softplus = lambda x: self._epsilon + softplus(x)
        self._sigmoid = torch.nn.Sigmoid()

        # Construct bias MLP
        self.bias_mlp = embedding_mlp.EmbeddingMLP(
            n_units=self.dataset.n_units,
            condition_on_position=False,
            condition_on_identity=False,
            **embedding_mlp_kwargs,
        )

        # Initialize the bias on the MLP's last layer
        mean_neural = torch.from_numpy(
            (
                np.nansum(self.dataset.train_data["neural"], axis=(0, 1))
                / np.nansum(
                    self.dataset.train_data["neural_finite"], axis=(0, 1)
                )
            ).astype(np.float32)
        )
        init_bias = (
            torch.log(torch.exp(self._softplus_beta * mean_neural) - 1)
            / self._softplus_beta
        )
        with torch.no_grad():
            self.bias_mlp.last_layer.bias.copy_(init_bias)

    @abc.abstractmethod
    def forward(self):
        """Forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def prediction_per_unit_numpy(self, train: bool) -> np.ndarray:
        """Get the model's prediction per unit as a numpy array.

        Returns:
            Numpy array of shape [n_trials, n_units, n_timesteps] containing the
            predicted firing rates for each unit across all trials and
            timesteps.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self) -> torch.Tensor:
        """Compute the loss for the model.

        Returns:
            A scalar tensor representing the loss.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _cache(self, write_dir: str):
        """Cache attention to disk.

        Args:
            write_dir: Directory to write the cached attention.
        """
        raise NotImplementedError

    def cache(self, write_dir: str):
        self.dataset.cache(write_dir / "dataset_cache")
        self._cache(write_dir / "model_cache")

    def goodness_of_fit_array(
        self,
        train: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the goodness of fit for the model.

        Args:
            train: Boolean indicating whether to compute for training data
                (True) or testing data (False).

        Returns:
            log_likelihood: Numpy array of shape (num_trials, num_timesteps, num_neurons)
                containing the log likelihood values.
            squared_error: Numpy array of shape (num_trials, num_timesteps, num_neurons)
                containing the squared error values.
            target_finite: Numpy array of shape (num_trials, num_timesteps, num_neurons)
                indicating whether the target values are finite.
        """
        prediction, neural, neural_finite = self.prediction_per_unit_numpy(
            train=train
        )
        squared_error = np.square(neural - prediction)
        log_likelihood = poisson_log_likelihood(
            neural, prediction, self._epsilon
        )

        return log_likelihood, squared_error, neural_finite

    def goodness_of_fit(self, train: bool = True) -> dict:
        """Compute the goodness of fit for the model.

        Args:
            train: Boolean indicating whether to compute for training data
                (True) or testing data (False).

        Returns:
            A dictionary containing the mean log likelihood, mean squared error,
                number of neuron trials, number of neurons, and number of
                trials.
        """
        log_likelihood, squared_error, neural_finite = (
            self.goodness_of_fit_array(train=train)
        )
        num_neuron_trials = np.sum(neural_finite)
        mll = np.sum(neural_finite * log_likelihood) / num_neuron_trials
        mse = np.sum(neural_finite * squared_error) / num_neuron_trials
        outputs = {
            "mean_log_likelihood": float(mll),
            "mean_squared_error": float(mse),
            "num_neuron_trials": int(num_neuron_trials),
            "num_neurons": int(self.dataset.n_units),
            "num_trials": int(self.dataset.n_trials),
        }
        return outputs

    @property
    @abc.abstractmethod
    def per_trial_parameters(self) -> list:
        """Get the parameters for per-trial variables."""
        raise NotImplementedError

    @abc.abstractmethod
    def _embedding_parameters(self) -> list:
        """Get the parameters for embedding variables."""
        raise NotImplementedError

    @property
    def embedding_parameters(self) -> list:
        """Get the parameters for embedding variables."""
        # return self._embedding_parameters() + list(self.bias_mlp.parameters())
        return self._embedding_parameters()

    @property
    def dataset(self):
        """Get the dataset."""
        return self._dataset
