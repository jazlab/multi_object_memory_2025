"""Gain model."""

import numpy as np
import torch

from . import abstract_model, embedding_mlp


class Gain(abstract_model.AbstractModel):
    """Gain model."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        condition_on_identity: bool = True,
        time_noise_sigma: float = 0.2,
        position_noise_sigma: float = 0.2,
        sigmoid_attention: bool = True,
        **embedding_mlp_kwargs
    ):
        """Constructor.

        Args:
            dataset: Dataset object containing neural data and stimuli. Must be
                an instance of phys_modeling.training.dataset.Dataset.
            batch_size: Size of the training batch.
            condition_on_identity: Boolean indicating whether to condition on
                identity in the embedding MLP.
            time_noise_sigma: Standard deviation of noise for time input.
            position_noise_sigma: Standard deviation of noise for position
                input.
            sigmoid_attention: Boolean indicating whether to use sigmoid
                nonlinearity for attention weights. If False, uses softplus.
            embedding_mlp_kwargs: Dictionary of keyword arguments for the
                embedding MLP. If None, defaults will be used.
        """
        super(Gain, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            time_noise_sigma=time_noise_sigma,
            position_noise_sigma=position_noise_sigma,
            **embedding_mlp_kwargs,
        )
        self._sigmoid_attention = sigmoid_attention

        # Construct attention weights
        attn_init = torch.zeros(
            (self._dataset.n_trials, self._dataset.max_n_objects)
        )
        self._attention = torch.nn.Parameter(attn_init, requires_grad=True)

        # Construct nonlinearities for attention and embedding
        if self._sigmoid_attention:
            self._attention_nonlinearity = torch.nn.Sigmoid()
        else:
            self._attention_nonlinearity = torch.nn.Softplus()

        # Construct embedding MLP
        self.embedding_mlp = embedding_mlp.EmbeddingMLP(
            n_units=self.dataset.n_units,
            condition_on_position=True,
            condition_on_identity=condition_on_identity,
            **embedding_mlp_kwargs,
        )

    def forward(
        self, train, batch_size=None, noise: bool = True
    ) -> torch.Tensor:
        """Forward pass of the model.

        Returns:
            A tensor of shape [n_trials, n_units, n_timesteps] containing the
            predicted firing rates for each unit across all trials and
            timesteps.
        """
        # Get a batch of training data of shape
        # [batch_size, n_objects, n_features]
        data = self.dataset.get_batch(train=train, batch_size=batch_size)
        trial = data["trial"]
        time = data["time"]
        positions = data["positions"] - 0.5
        identities = data["identities"]
        neural = data["neural"]
        neural_finite = data["neural_finite"]

        # Add noise to time and position
        if noise:
            time += torch.randn_like(time) * self._time_noise_sigma
            positions += (
                torch.randn_like(positions) * self._position_noise_sigma
            )

        # [batch_size, n_objects, n_units]
        bias = self.bias_mlp(time=time)

        # [batch_size, n_objects, n_units]
        reps = [1] * len(time.shape) + [self._dataset.max_n_objects]
        times_tiled = time.unsqueeze(-1).repeat(reps)
        embeddings = self.embedding_mlp(
            time=times_tiled,
            position=positions,
            identity=identities,
        )

        # [batch_size, n_objects]
        attention = self.get_attention_matrix()[trial]

        # [batch_size, n_units]
        prediction = self._softplus(
            bias + torch.sum(embeddings * attention[..., None], dim=-2)
        )

        return prediction, neural, neural_finite

    def prediction_per_unit_numpy(self, train: bool) -> np.ndarray:
        """Get the model's prediction per unit as a numpy array.

        Args:
            train: Boolean indicating whether to use training or validation
                data.

        Returns:
            Numpy array of shape [n_trials, n_timesteps, n_units] containing the
            predicted firing rates for each unit across all trials and
            timesteps.
        """
        prediction, neural, neural_finite = self.forward(
            train=train, batch_size=None, noise=False
        )
        prediction = prediction.detach().cpu().numpy()
        neural = neural.detach().cpu().numpy()
        neural_finite = neural_finite.detach().cpu().numpy()

        return prediction, neural, neural_finite

    def loss(self) -> torch.Tensor:
        """Compute the training loss for the model.

        Returns:
            A scalar tensor representing the loss.
        """
        prediction, neural, neural_finite = self.forward(
            train=True, batch_size=self._batch_size, noise=True
        )
        negative_log_likelihood = prediction - neural * torch.log(prediction)
        loss = torch.sum(neural_finite * negative_log_likelihood)
        loss /= torch.sum(neural_finite)
        return loss

    def _cache(self, write_dir: str):
        """Cache attention to disk.

        Args:
            write_dir: Directory to write the cached attention.
        """
        write_dir.mkdir(parents=True, exist_ok=True)
        attention_matrix = self.get_attention_matrix().detach().numpy()
        attention_path = write_dir / "attention.npy"
        np.save(attention_path, attention_matrix)
        return

    def get_attention_matrix(self) -> torch.Tensor:
        """Get the attention matrix.

        Returns:
            A tensor of shape [n_trials, n_objects] containing the attention
            weights for each trial and object.
        """
        attention = (
            self.dataset.valid_objects_torch
            * self._attention_nonlinearity(self._attention)
        )
        return attention

    def _embedding_parameters(self) -> list:
        """Get the parameters for embedding variables."""
        return list(self.embedding_mlp.parameters())

    @property
    def per_trial_parameters(self) -> list:
        """Get the parameters for per-trial variables."""
        return [self._attention]
