import numpy as np
import torch
import math


def extract_impedance(rho, vp):
    return rho * vp


def extract_reflectivity(z):
    zi = z[:-1]
    zii = z[1:]
    return (zii - zi) / (zii + zi)


def extract_seismic(r, w):
    """Convolução entre refletividade `r` e wavelet `w`.

    Aceita `r` e `w` como `numpy.ndarray`, `torch.Tensor` (CPU ou CUDA) ou listas.
    Se receber tensores PyTorch em GPU, move-os para CPU antes de converter para NumPy.

    Retorna um `numpy.ndarray` resultante da convolução (comportamento equivalente a `np.convolve`).
    """
    # Se for tensor PyTorch, converte para numpy em CPU
    if isinstance(r, torch.Tensor):
        r = r.detach().cpu().numpy()
    if isinstance(w, torch.Tensor):
        w = w.detach().cpu().numpy()

    # Garante arrays NumPy 1D
    r = np.asarray(r).ravel()
    w = np.asarray(w).ravel()

    return np.convolve(r, w)


def distort_tdr(tdr, sigma=5, scale=10):
    noise = np.convolve(
        np.random.randn(len(tdr)),
        np.exp(-(np.linspace(-2, 2, sigma) ** 2)),
        mode="same",
    )
    noise = noise / np.max(np.abs(noise))
    shift = noise * scale

    tdr_distorted = tdr + shift
    return tdr_distorted, shift


def add_awgn(s, snr_db):
    s_power = np.mean(s**2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = s_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), size=s.shape)
    noisy_s = s + noise

    return noisy_s, noise


def generate_distort_tdr(tdr, n):
    tdrs = torch.tensor(())
    shifts = torch.tensor(())
    for i in range(n):
        tdr_d, shift = distort_tdr(tdr)
        tdrs = torch.cat(tdrs, tdr_d)
        shifts = torch.cat(shifts, shift)

    return tdrs, shifts


def ricker_wavelet(f, dt, nt):
    """Ricker (Mexican hat) wavelet"""
    t = (torch.arange(nt) - (nt - 1) / 2) * dt
    pi2 = (math.pi * f) ** 2
    w = (1 - 2 * pi2 * t**2) * torch.exp(-pi2 * t**2)

    return w / torch.max(torch.abs(w))


def gabor_wavelet(f, dt, nt):
    """Gabor wavelet (Gaussian modulated cosine)"""
    t = (torch.arange(nt) - (nt - 1) / 2) * dt
    sigma = 1.0 / (2 * math.pi * f)
    w = torch.exp(-(t**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * f * t)
    return w / torch.max(torch.abs(w))


def ormsby_wavelet(f1, f2, f3, f4, dt, nt):
    """Ormsby wavelet (bandpass trapezoidal)"""
    t = (torch.arange(nt) - (nt - 1) / 2) * dt

    def sinc(f, t):
        return torch.sinc(2 * f * t)

    w = (
        f4**2 * sinc(f4, t) ** 2
        - f3**2 * sinc(f3, t) ** 2
        - f2**2 * sinc(f2, t) ** 2
        + f1**2 * sinc(f1, t) ** 2
    ) / (f4 - f3 - f2 + f1)

    return w / torch.max(torch.abs(w))


# def klauder_wavelet(f1, f2, T, dt, nt=97):
#     """Klauder wavelet (chirp-like sweep autocorrelation)"""
#     ns = int(round(T / dt))
#     t_sweep = (torch.arange(ns) - ns / 2) * dt

#     # Chirp linear
#     f0 = 0.5 * (f1 + f2)
#     k = (f2 - f1) / T
#     sweep = torch.cos(2 * math.pi * (f1 * t_sweep + 0.5 * k * t_sweep**2))
#     # Janela (opcional, mas recomendado)
#     window = torch.hann_window(len(t_sweep))
#     sweep = sweep * window
#     sweep = sweep / torch.sqrt(torch.sum(sweep**2))

#     # Autocorrelação usando correlação cruzada
#     autocorr = torch.nn.functional.conv1d(
#         sweep.view(1, 1, -1), sweep.flip(0).view(1, 1, -1), padding=len(sweep) - 1
#     ).flatten()

#     # Extrair parte central
#     center = len(autocorr) // 2
#     half = nt // 2
#     w = (
#         autocorr[center - half : center + half + 1]
#         if nt % 2
#         else autocorr[center - half : center + half]
#     )

#     return w / w.abs().max()


def klauder_wavelet(f1, f2, T, dt, nt=97):
    """Klauder wavelet (chirp autocorrelation)"""

    ns = int(round(T / dt))
    t_sweep = np.arange(ns) * dt - T / 2

    # Chirp linear
    k = (f2 - f1) / T
    phase = 2 * np.pi * (f1 * t_sweep + 0.5 * k * t_sweep**2)
    sweep = np.cos(phase)

    # Janela
    window = np.hanning(ns)
    sweep = sweep * window
    sweep = sweep / np.sqrt(np.sum(sweep**2))

    # Autocorrelação usando numpy (mais confiável)
    autocorr = np.correlate(sweep, sweep, mode="full")

    # Extrair centro
    center = len(autocorr) // 2
    start = center - nt // 2
    w = autocorr[start : start + nt]

    # Converter de volta para torch
    w = torch.from_numpy(w).float()

    return w / torch.abs(w).max()


def sinc_wavelet(f, dt, nt):
    """Sinc wavelet (band-limited impulse)"""
    t = torch.linspace(-nt // 2, nt // 2, nt) * dt
    w = torch.where(
        t == 0,
        torch.tensor(1.0, device=t.device),
        torch.sin(2 * math.pi * f * t) / (2 * math.pi * f * t),
    )
    return w / torch.max(torch.abs(w))


def power_spectrum(a):
    a = torch.tensor(a)
    af = torch.fft.rfft(a, dim=-1)

    ap = torch.abs(af) ** 2

    ap = torch.clamp(ap, min=1e-8)

    ap = ap / (torch.sum(ap, dim=-1, keepdim=True) + 1e-8)

    return ap


__all__ = [
    "extract_impedance",
    "extract_reflectivity",
    "extract_seismic",
    "distort_tdr",
    "add_awgn",
    "generate_distort_tdr",
    "ricker_wavelet",
    "gabor_wavelet",
    "ormsby_wavelet",
    "klauder_wavelet",
    "sinc_wavelet",
    "power_spectrum",
]
