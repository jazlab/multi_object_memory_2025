"""Switching model."""

import numpy as np
import torch

from . import abstract_model, embedding_mlp, smoothing


class Switching(abstract_model.AbstractModel):
    """Base class for switching models."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        smoothness_attention_ms: float,
        condition_on_identity: bool = True,
        time_noise_sigma: float = 0.2,
        position_noise_sigma: float = 0.2,
        **embedding_mlp_kwargs
    ):
        """Constructor.

        Args:
            dataset: Dataset object containing neural data and stimuli. Must be
                an instance of phys_modeling.training.dataset.Dataset.
            batch_size: Size of the training batch.
            smoothness_attention_ms: Smoothness of attention in milliseconds.
            condition_on_identity: Boolean indicating whether to condition on
                identity in the embedding MLP.
            time_noise_sigma: Standard deviation of noise for time input.
            position_noise_sigma: Standard deviation of noise for position
                input.
            embedding_mlp_kwargs: Dictionary of keyword arguments for the
                embedding MLP. If None, defaults will be used.
        """
        super(Switching, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            time_noise_sigma=time_noise_sigma,
            position_noise_sigma=position_noise_sigma,
            **embedding_mlp_kwargs,
        )
        self._smoothness_attention_ms = smoothness_attention_ms

        # Construct attention weights
        n_slots = 1 + self.dataset.max_n_objects
        attn_init = torch.zeros(
            (self.dataset.n_trials, n_slots, self.dataset.n_timesteps)
        )
        self._attention = torch.nn.Parameter(attn_init, requires_grad=True)

        # Construct smoothing kernels and normalizations
        self._kernel_attention, self._normalization_attention = (
            smoothing.get_torch_kernel_and_normalization(
                self._smoothness_attention_ms, self.dataset.n_timesteps
            )
        )

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

        Args:
            train: Boolean indicating whether to use training data or test data.
            batch_size: Size of the batch to use for training. If None, uses the
                default batch size from the dataset.
            noise: Boolean indicating whether to add noise to time and position.

        Returns:
            prediction_per_object: Tensor of shape
                [batch_size, n_objects + 1, n_units].
            attention_per_object: Tensor of shape
                [batch_size, n_objects + 1, n_timesteps].
            neural: Tensor of shape [batch_size, n_units].
            neural_finite: Tensor of shape [batch_size, n_units].
        """
        # Get a batch of training data of shape
        # [batch_size, n_objects, n_features]
        data = self.dataset.get_batch(train=train, batch_size=batch_size)
        trial = data["trial"]
        time = data["time"]
        time_indices = data["time_indices"]
        positions = data["positions"] - 0.5
        identities = data["identities"]
        neural = data["neural"]
        neural_finite = data["neural_finite"]

        # Add noise to time and position if needed
        if noise:
            time += torch.randn_like(time) * self._time_noise_sigma
            positions += (
                torch.randn_like(positions) * self._position_noise_sigma
            )

        # [batch_size, n_objects, n_units]
        bias = self.bias_mlp(time=time)

        # [batch_size, n_objects, n_units]
        reps = [1] * len(time.shape) + [self._dataset.max_n_objects]
        time_tiled = time.unsqueeze(-1).repeat(reps)
        embeddings = self.embedding_mlp(
            time=time_tiled,
            position=positions,
            identity=identities,
        )
        # Add bias-only embedding for null slot
        null_embedding = torch.zeros(
            embeddings.shape[:-2] + (1, embeddings.shape[-1]),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        embeddings = torch.cat([embeddings, null_embedding], dim=-2)

        # [batch_size, n_objects + 1, n_units]
        prediction_per_object = self._softplus(bias.unsqueeze(-2) + embeddings)

        # [n_trials, n_timesteps, n_objects]
        attention = self.get_attention_matrix()
        attention_per_object = attention[trial, time_indices]

        return (
            prediction_per_object,
            attention_per_object,
            neural,
            neural_finite,
        )

    def prediction_per_unit_numpy(self, train) -> np.ndarray:
        """Get the model's prediction per unit as a numpy array.

        Returns:
            prediction: Numpy array of shape [n_trials, n_timesteps, n_units]
                containing the predicted firing rates for each unit across all
                trials and timesteps.
            neural: Numpy array of shape [n_trials, n_timesteps, n_units].
            neural_finite: Numpy array of shape [n_trials, n_timesteps, n_units].
        """
        # prediction has shape [n_trials, n_timesteps, n_objects, n_units]
        # attention has shape [n_trials, n_timesteps, n_objects]
        # neural and neural_finite have shape [n_trials, n_timesteps, n_units]
        prediction, attention, neural, neural_finite = self.forward(
            train=train, batch_size=None, noise=False
        )

        # max_attn has shape [n_trials, n_timesteps]
        max_attn = torch.argmax(attention, axis=-1)

        # Index the prediction tensor to get the predicted firing rates
        n_trials, n_timesteps, _, _ = prediction.shape
        trials_index = (
            torch.arange(n_trials)
            .view(n_trials, 1)
            .expand(n_trials, n_timesteps)
        )
        timesteps_index = (
            torch.arange(n_timesteps)
            .view(1, n_timesteps)
            .expand(n_trials, n_timesteps)
        )
        prediction = prediction[trials_index, timesteps_index, max_attn]

        # Detach and convert to numpy
        prediction = prediction.detach().cpu().numpy()
        neural = neural.detach().cpu().numpy()
        neural_finite = neural_finite.detach().cpu().numpy()

        return prediction, neural, neural_finite

    def loss(self) -> torch.Tensor:
        """Compute the training loss for the model.

        Returns:
            A scalar tensor representing the loss.
        """
        # prediction has shape [batch_size, n_objects, n_units]
        # attention has shape [batch_size, n_objects]
        # neural and neural_finite have shape [batch_size, n_units]
        prediction, attention, neural, neural_finite = self.forward(
            train=True, batch_size=self._batch_size, noise=True
        )
        negative_log_likelihood = prediction - neural[:, None] * torch.log(
            prediction
        )
        loss_matrix = neural_finite[:, None] * negative_log_likelihood

        # Sum loss over units
        loss_matrix = torch.sum(loss_matrix, axis=2)

        # Weight loss by attention and normalize
        loss = torch.sum(loss_matrix * attention) / torch.sum(neural_finite)

        return loss

    def _cache(self, write_dir: str):
        """Cache attention to disk.

        Args:
            write_dir: Directory to write the cached attention.
        """
        write_dir.mkdir(parents=True, exist_ok=True)
        attention = self.get_attention_matrix()
        max_attn = torch.argmax(attention, axis=2)
        max_attn = max_attn.detach().numpy()
        null_value = np.max(max_attn)
        attention_matrix = np.stack(
            [np.mean(max_attn == i, axis=1) for i in range(null_value)], axis=1
        )
        attention_path = write_dir / "attention.npy"
        np.save(attention_path, attention_matrix)
        return

    def get_attention_matrix(self) -> torch.Tensor:
        """Get attention matrix.

        Returns:
            A tensor of shape [n_trials, n_timesteps, n_objects] containing the
            attention weights.
        """
        # Get attention per trial
        keep_attention = torch.cat(
            [
                self.dataset.valid_objects_torch,
                torch.ones((self.dataset.n_trials, 1)),
            ],
            axis=1,
        )
        exp_attn = torch.exp(self._attention)
        exp_attn_masked = exp_attn * keep_attention[:, :, None]
        softmax_attn = exp_attn_masked / torch.sum(
            exp_attn_masked, axis=1, keepdim=True
        )

        # Smooth attention
        softmax_attn = smoothing.smooth(
            softmax_attn, self._kernel_attention, self._normalization_attention
        )
        softmax_attn = softmax_attn.transpose(1, 2)

        return softmax_attn

    def _embedding_parameters(self) -> list:
        """Get parameters for embedding variables."""
        return list(self.embedding_mlp.parameters())

    @property
    def per_trial_parameters(self) -> list:
        """Get parameters for per-trial variables."""
        return [self._attention]
