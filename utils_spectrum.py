import torch
from torch import fft
import numpy as np
from typing import Union, Tuple
import torch.nn.functional as F

def get_fft(
    signal: Union[torch.Tensor, np.ndarray, list],
    duration: float,
    dt: float,
) -> torch.Tensor:
    
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    N = int(duration / dt)
    if N % 2 != 0:
        N += 1  # Make even
    spectrum = torch.abs(fft.rfft(signal, n=N, norm=None))
    return spectrum


def get_amplitude_spectra(

    signal: Union[torch.Tensor, np.ndarray, list],
    duration: float,
    dt: float,

) -> torch.Tensor:

    """
    Calculates the amplitude spectrum of a signal.
    Args:
        signal (Union[torch.Tensor, np.ndarray, list]): The input signal.
            If not a torch.Tensor, it will be converted.
        duration (float): The duration of the signal in seconds.
        dt (float): The sampling interval in seconds.
    Returns:
        amplitudes (torch.Tensor): The amplitude for each frequency bin.
    """

    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    N = int(duration / dt)
    if N % 2 != 0:
        N += 1  # Make even
    spectrum = torch.abs(fft.rfft(signal, n=N, norm=None))
    amplitudes = spectrum / N
    amplitudes[..., 1:-1] *= 2
    return amplitudes

def get_freqs(duration, dt):
    N = int(duration / dt)
    if N % 2 != 0:
        N += 1
    freqs = fft.rfftfreq(N, d=float(dt))

    return freqs

def pow2db(power: Union[torch.Tensor, np.ndarray, list], normalize: bool = False) -> torch.Tensor:

    """
    Converts a power value to decibels (dB).

    Args:
        power (Union[torch.Tensor, np.ndarray, list]): The input power values.
            If not a torch.Tensor, it will be converted.
        normalize (bool, optional): Whether to normalize the power before
            converting to dB. Defaults to False.
    Returns:
        torch.Tensor: The power values in dB.
    """

    if not isinstance(power, torch.Tensor):
        power = torch.tensor(power, dtype=torch.float32)

    eps = torch.tensor(1e-12)
    if normalize:
        power = power / (torch.max(power) + eps)
    return 20 * torch.log10(power + eps)

def get_power_spectra(
    signal: Union[torch.Tensor, np.ndarray, list],
    duration: float,
    dt: float,
) -> torch.Tensor:

    """
    Calculates the power spectrum of a signal.
    Args:
        signal (Union[torch.Tensor, np.ndarray, list]): The input signal.
            If not a torch.Tensor, it will be converted.
        duration (float): The duration of the signal in seconds.
        dt (float): The sampling interval in seconds.

    Returns:
        power_spectrum (torch.Tensor): The power for each frequency bin.
    """

    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    N = int(duration / dt)
    if N % 2 != 0:
        N += 1  # Make even
    spectrum = torch.abs(fft.rfft(signal, n=N, norm=None))
    power_spectrum = (spectrum / N) ** 2
    power_spectrum[..., 1:-1] *= 2
    return power_spectrum


def moving_average(
    data: Union[torch.Tensor, np.ndarray, list],
    window_size: int
) -> torch.Tensor:
    """
    Applies a moving average filter to the input data, returning an output
    tensor of the same size as the input.

    Args:
        data (Union[torch.Tensor, np.ndarray, list]): The input data.
            If not a torch.Tensor, it will be converted.
        window_size (int): The size of the moving average window. Must be a positive integer.

    Returns:
        torch.Tensor: The data after applying the moving average, with the same size as the input.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if window_size > data.shape[-1]:
        raise ValueError("window_size cannot be larger than the input data length.")

    # Reshape data for conv1d: (batch, channels, length)
    original_dim = data.dim()
    if original_dim == 1:
        data_reshaped = data.unsqueeze(0).unsqueeze(0) # -> (1, 1, length)
    elif original_dim == 2:
        data_reshaped = data.unsqueeze(0) # -> (1, channels, length)
    else:
        raise ValueError("Input data must be 1D or 2D.")

    # Calculate padding to maintain same output size
    total_padding = window_size - 1
    pad_left = total_padding // 2
    pad_right = total_padding - pad_left

    # Apply padding
    # F.pad expects (padding_left, padding_right, padding_top, padding_bottom, ...)
    # For 1D signal (batch, channels, length), we pad on the 'length' dimension
    padded_data = F.pad(data_reshaped, (pad_left, pad_right), mode='replicate')

    in_channels = data_reshaped.shape[1]
    kernel = torch.ones(in_channels, 1, window_size, device=data.device, dtype=data.dtype) / window_size

    # Apply 1D convolution
    averaged_data = F.conv1d(padded_data, kernel, groups=in_channels)

    # Reshape back to original dimensions
    if original_dim == 1:
        return averaged_data.squeeze(0).squeeze(0)
    else: # original_dim == 2
        return averaged_data.squeeze(0)
