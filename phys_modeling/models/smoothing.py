"""Smoothing tools."""

import math

import constants
import numpy as np
import torch


def get_smoothing_kernel(smoothness_ms: float) -> np.ndarray:
    """Get smoothing kernel.

    Args:
        smoothness_ms: Float indicating the desired smoothness in milliseconds.

    Returns:
        Numpy array of the smoothing kernel. The kernel is triangular and
        symmetric and has a total width of 2 * smoothness_ms.
    """
    bin_size_ms = constants.SPIKE_COUNT_BIN_SIZE_MS
    kernel_half_width = math.ceil(float(smoothness_ms) / bin_size_ms)
    distance_from_center = bin_size_ms * np.arange(kernel_half_width)
    half_kernel = smoothness_ms - distance_from_center
    if half_kernel[-1] < 0:
        half_kernel = half_kernel[:-1]
    kernel = np.concatenate((half_kernel[::-1], half_kernel[1:])).astype(float)
    kernel /= np.sum(kernel)
    return kernel


def get_torch_kernel_and_normalization(
    smoothness_ms: float,
    n_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get smoothing kernel and normalization for torch.

    Args:
        smoothness_ms: Float indicating the desired smoothness in milliseconds.
        n_timesteps: Integer indicating the number of timesteps in the data.

    Returns:
        kernel_torch: Torch tensor of the smoothing kernel with shape
            [1, 1, n_timesteps].
        conv_ones: Torch tensor of the normalization with shape
            [1, n_timesteps].
    """
    kernel = get_smoothing_kernel(smoothness_ms)
    kernel_torch = torch.from_numpy(kernel.astype(np.float32))
    kernel_torch = kernel_torch[None, None]
    ones_input = torch.ones((1, 1, n_timesteps))
    conv_ones = torch.nn.functional.conv1d(
        ones_input, kernel_torch, padding="same"
    )[:, 0]
    return kernel_torch, conv_ones


def smooth(
    x: torch.Tensor,
    kernel: torch.Tensor,
    conv_ones: torch.Tensor,
) -> torch.Tensor:
    """Smooth a tensor over time using convolution.

    Args:
        x: Torch tensor of shape [..., n_timesteps] to smooth.
        kernel: Torch tensor of shape [1, 1, n_timesteps] representing the
            smoothing kernel.
        conv_ones: Torch tensor of shape [1, n_timesteps] representing the
            normalization factor for the convolution.

    Returns:
        Torch tensor of the smoothed input with the same shape as x.
    """
    n_timesteps = conv_ones.shape[1]
    x_shape = x.shape
    x_input = x.view(-1, 1, n_timesteps)
    conv = torch.nn.functional.conv1d(x_input, kernel, padding="same")
    conv = conv[:, 0] / conv_ones
    conv = torch.reshape(conv, x_shape)
    return conv
