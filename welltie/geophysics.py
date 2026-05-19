import numpy as np
from scipy.interpolate import CubicSpline
import torch
import math

from scipy.interpolate import interp1d

def extract_impedance(rho, vp):
    return rho * vp


def extract_reflectivity(z):
    r = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    return np.concatenate((r, [r[-1]]))


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

    return np.convolve(r, w, mode="same")


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

    if s_power == 0:
        return s, np.zeros_like(s)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = s_power / snr_linear
    noise_std = np.sqrt(max(noise_power, 1e-12))
    noise = np.random.normal(0, noise_std, size=s.shape)
    noisy_s = s + noise

    return noisy_s.astype(np.float32), noise.astype(np.float32)


def generate_twt(sonic, depth, start_twt=0.0):
    sonic = np.asarray(sonic)
    depth = np.asarray(depth)

    dx = np.diff(depth)
    #sonic_mean = 0.5 * (sonic[:-1] + sonic[1:])
    dt = sonic[:-1] * dx # sonic[:-1]
    t = np.concatenate(([0], np.cumsum(dt))) * 2e-6
    t = t + start_twt

    return t

def generate_warped_twt(tdr, max_shift=20, smoothness=50, n_knots=12):
    n = len(tdr)
    knot_idx = np.linspace(0, n-1, n_knots).astype(int)
    knot_t = tdr[knot_idx]

    # 1. Generate a bulk static shift (simulate datum/KB errors)
    bulk_shift = np.random.uniform(-max_shift/2, max_shift/2)

    # 2. Generate random variations
    knot_shift = np.random.uniform(-max_shift/2, max_shift/2, size=n_knots)
    
    # 50% chance to sort the shifts to create a realistic velocity drift trend
    if np.random.rand() > 0.5:
        knot_shift = np.sort(knot_shift) 
        if np.random.rand() > 0.5:
            knot_shift = knot_shift[::-1] # Sometimes drift the other way

    # 3. Add the bulk shift to the variations
    knot_shift += bulk_shift

    # 4. Smooth interpolation
    cs = CubicSpline(knot_t, knot_shift, bc_type='natural')
    shift = cs(tdr)

    # 5. Enforce monotonic Tw (prevent time going backwards)
    tdr_warped = tdr + shift
    tdr_warped = np.maximum.accumulate(tdr_warped)

    shift = tdr_warped - tdr

    return tdr_warped, shift

def resample_logs_to_seismic(twt, log, dt, method='backus'):
    """
    method : str
        'linear': Interpolação linear simples (rápido, mas pode causar aliasing).
        'block': Média aritmética em blocos (melhor para densidade).
        'backus': (Simplificado) Média harmônica para Vp (fisicamente correto para ondas).
    """

    t_min = np.ceil(twt.min() / dt) * dt
    t_max = np.floor(twt.max() / dt) * dt
    axis = np.arange(t_min, t_max + dt, dt)

    def process_arr(arr, method_name):
        if method_name == 'linear':
            f = interp1d(twt, arr, kind='linear', fill_value="extrapolate")
            return f(axis)
        elif method_name in ['block', 'backus']:
            out_arr = np.zeros_like(axis)
            for i, t in enumerate(axis):
                mask = (twt >= t - dt/2) & (twt < t + dt/2)

                if np.sum(mask) > 0:
                    vals = arr[mask]
                    if method_name == 'backus':
                        out_arr[i] = 1 / np.mean(1 / vals)
                    else:
                        out_arr[i] = np.mean(vals)
                else:
                    if i > 0: out_arr[i] = out_arr[i - 1]
                    else: out_arr[i] = arr[0]
        return out_arr

    if isinstance(log, dict):
        resampled_logs = {}
        for key, val in log.items():
            current_method = method
            if method == 'backus' and key.lower() != 'vp':
                current_method = 'block'
            
            resampled_logs[key] = process_arr(val, current_method)
        return axis, resampled_logs
    return axis, process_arr(log, method)

def depth2time_interpolation(x, tdr, dt=0.002):
    t_min = np.floor(tdr.min() / dt) * dt
    t_max = np.ceil(tdr.max() / dt) * dt
    axis = np.arange(t_min, t_max, dt)

    log = np.interp(axis, tdr, x)
    
    return log, axis

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