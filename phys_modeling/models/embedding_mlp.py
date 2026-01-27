"""Gain model."""

import abc

import numpy as np
import torch


class EmbeddingMLP(torch.nn.Module):
    """MLP class embedding functions."""

    def __init__(
        self,
        n_units: int,
        hidden_sizes: tuple = (512, 512),
        condition_on_position: bool = True,
        condition_on_identity: bool = True,
        identity_features: int = 3,
        time_frequencies: tuple = (1, 2, 4, 8),
        position_frequencies: tuple = (1, 2, 4, 8),
        activation=None,
        bias=True,
        activate_final=False,
    ):
        """Constructor.

        Args:
            n_units: Number of units in the model.
            condition_on_position: Whether to include spatial input.
            condition_on_identity: Whether to include identity input.
            time_frequencies: Frequencies for time input features.
            position_frequencies: Frequencies for position input features.
            hidden_sizes: Sizes of hidden layers in the MLP.
            activation: Activation function to use in the MLP. If None, defaults
                to ReLU.
            bias: Whether to include bias in the MLP layers.
            activate_final: Whether to apply activation to the final layer.
        """
        super().__init__()
        self._n_units = n_units
        self._hidden_sizes = hidden_sizes
        self._condition_on_position = condition_on_position
        self._condition_on_identity = condition_on_identity
        self._identity_features = identity_features
        self._time_frequencies = time_frequencies
        self._position_frequencies = position_frequencies
        self._bias = bias
        self._activate_final = activate_final

        # Compute activation function
        if activation is None:
            activation = torch.nn.ReLU()
        self._activation = activation

        # Compute input features
        self._in_features = 1 + len(self._time_frequencies)
        if self._condition_on_position:
            self._in_features += 2 + 2 * len(self._position_frequencies)
        if self._condition_on_identity:
            self._in_features += self._identity_features

        # Compute layer sizes
        features_list = (
            [self._in_features] + list(self._hidden_sizes) + [self._n_units]
        )

        # Make the MLP
        module_list = []
        for i in range(len(features_list) - 1):
            if i > 0:
                module_list.append(self._activation)
            layer = torch.nn.Linear(
                in_features=features_list[i],
                out_features=features_list[i + 1],
                bias=self._bias,
            )
            module_list.append(layer)
        self._last_layer = layer

        if self._activate_final:
            module_list.append(self._activation)

        self.net = torch.nn.Sequential(*module_list)

    def forward(self, time, position=None, identity=None):
        """Apply MLP to input.

        Args:
            time: Tensor of shape [batch_size, ...].
            position: Tensor of shape [batch_size, ..., 2].
            identity: Tensor of shape [batch_size, ..., identity_features].
            add_noise: Whether to add noise to the input features.

        Returns:
            Output of shape [batch_size, ..., self.out_features]. If
                self._apply_to_last_dim, then an arbitrary number of
                intermediate dimensions will be preserved.
        """
        # Prepare time input tensor
        time_inputs = [time] + [
            torch.sin(f * time) for f in self._time_frequencies
        ]
        time_inputs = torch.stack(time_inputs, dim=-1)
        mlp_inputs = [time_inputs]

        # Prepare position input tensor
        if self._condition_on_position and position is not None:
            position_inputs = [position] + [
                torch.sin(f * position) for f in self._position_frequencies
            ]
            position_inputs = torch.concat(position_inputs, dim=-1)
            mlp_inputs.append(position_inputs)

        # Prepare identity input tensor
        if self._condition_on_identity and identity is not None:
            mlp_inputs.append(identity)

        # Combine inputs
        mlp_inputs = torch.concat(mlp_inputs, dim=-1)

        # Get outputs
        return self.net(mlp_inputs)

    @property
    def n_units(self):
        """Number of units in the model."""
        return self._n_units

    @property
    def hidden_sizes(self):
        """Sizes of hidden layers in the MLP."""
        return self._hidden_sizes

    @property
    def in_features(self):
        """Number of input features to the MLP."""
        return self._in_features

    @property
    def condition_on_position(self):
        """Whether the model uses spatial input."""
        return self._condition_on_position

    @property
    def condition_on_identity(self):
        """Whether the model uses identity input."""
        return self._condition_on_identity

    @property
    def time_frequencies(self):
        """Frequencies used for time input features."""
        return self._time_frequencies

    @property
    def position_frequencies(self):
        """Frequencies used for position input features."""
        return self._position_frequencies

    @property
    def time_noise_sigma(self):
        """Standard deviation of noise for time input."""
        return self._time_noise_sigma

    @property
    def position_noise_sigma(self):
        """Standard deviation of noise for position input."""
        return self._position_noise_sigma

    @property
    def bias(self):
        """Whether the MLP layers include bias."""
        return self._bias

    @property
    def activate_final(self):
        """Whether to apply activation to the final layer."""
        return self._activate_final

    @property
    def last_layer(self):
        """Last layer of the MLP."""
        return self._last_layer
