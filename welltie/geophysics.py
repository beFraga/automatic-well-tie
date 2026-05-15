import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import torch
import math
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz

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

def add_awgn(s, snr_db):
    s_power = np.mean(s**2)

    if s_power == 0:
        return s, np.zeros_like(s)

    snr_linear = 10**(snr_db/10)
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
    """
    n_samples = len(tdr)

    # Step 1: generate smooth noise (same idea as your original)
    noise = np.random.normal(0, 1, n_samples)
    kernel = np.ones(smoothness) / smoothness
    smooth_noise = np.convolve(noise, kernel, mode='same')

    # Step 2: normalize noise
    smooth_noise = smooth_noise / np.max(np.abs(smooth_noise))

    # Step 3: convert noise into positive increments
    # base increment = average spacing of original tdr
    base_increment = np.mean(np.diff(tdr))

    # create small variations around 1.0
    increments = base_increment * (1 + smooth_noise * (max_shift_ms / base_increment))

    # ensure strictly positive increments (monotonic guarantee)
    epsilon = 1e-6
    increments[increments <= 0] = epsilon

    # Step 4: build warped time via cumulative sum
    tdr_warped = np.cumsum(increments)

    # Step 5: rescale to match original range
    tdr_warped = (tdr_warped - tdr_warped[0])
    tdr_warped = tdr_warped / tdr_warped[-1]  # normalize to [0,1]
    tdr_warped = tdr_warped * (tdr[-1] - tdr[0]) + tdr[0]

    # Step 6: compute effective shift (for compatibility with your original output)
    time_shift = tdr_warped - tdr

    return tdr_warped, time_shift
    """

    """
    n = len(tdr)
    dt = np.diff(tdr)
    dt = np.append(dt, dt[-1])  # match size

    # Step 1: generate Gaussian noise
    noise = np.random.normal(0, 1, n)

    # Step 2: smooth it
    noise = gaussian_filter1d(noise, sigma=smoothness)

    # Step 3: normalize
    noise /= np.max(np.abs(noise))

    # Step 4: convert to derivative perturbation
    # ensures: dTw/dt = 1 + alpha * noise > 0
    alpha = 0.8  # < 1 guarantees positivity
    derivative = 1 + alpha * noise

    # Step 5: build warped time via integration
    tdr_warped = np.zeros_like(tdr)
    tdr_warped[0] = tdr[0]

    for i in range(1, n):
        tdr_warped[i] = tdr_warped[i-1] + derivative[i] * dt[i]

    # Step 6: rescale shift amplitude to desired max_shift
    shift = tdr_warped - tdr
    shift = shift / np.max(np.abs(shift)) * max_shift
    tdr_warped = tdr + shift

    return tdr_warped, shift
    """


    """
    n = len(tdr)

    # --- 1. choose knot positions (coarse control)
    knot_idx = np.linspace(0, n-1, n_knots).astype(int)
    knot_t = tdr[knot_idx]

    # --- 2. generate smooth shift values at knots
    knot_shift = np.random.uniform(-max_shift, max_shift, size=n_knots)

    # enforce realistic trend (optional but recommended)
    knot_shift = np.sort(knot_shift)  # creates drift-like behavior

    # --- 3. interpolate smoothly
    cs = CubicSpline(knot_t, knot_shift, bc_type='natural')
    shift = cs(tdr)

    # --- 4. enforce monotonic Tw
    tdr_warped = tdr + shift
    tdr_warped = np.maximum.accumulate(tdr_warped)

    shift = tdr_warped - tdr

    return tdr_warped, shift
    """

    """
    n = len(tdr)

    # --- 1. knots
    knot_idx = np.linspace(0, n-1, n_knots).astype(int)
    knot_t = tdr[knot_idx]

    # --- 2. random shifts (NO sorting)
    knot_shift = np.random.uniform(-max_shift, max_shift, size=n_knots)

    # --- 3. smooth interpolation
    cs = CubicSpline(knot_t, knot_shift, bc_type='natural')
    shift = cs(tdr)

    # --- 4. remove global bias (center around zero)
    shift -= np.mean(shift)

    # --- 5. limit amplitude
    shift = shift / np.max(np.abs(shift)) * max_shift

    # --- 6. enforce monotonic Tw
    tdr_warped = tdr + shift
    tdr_warped = np.maximum.accumulate(tdr_warped)

    shift = tdr_warped - tdr

    return tdr_warped, shift
    """

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


def ricker_wavelet(f, dt, nt):
    """Ricker (Mexican hat) wavelet"""
    t = torch.linspace(-nt//2, nt//2, nt) * dt
    pi2 = (math.pi * f) ** 2
    w = (1 - 2 * pi2 * t**2) * torch.exp(-pi2 * t**2)
    return w / torch.max(torch.abs(w))

def ricker(f, length, dt):
    t0 = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t0**2)) * np.exp(-(np.pi**2)*(f**2)*(t0**2))
    return t0, y

def gabor_wavelet(f, dt, nt):
    """Gabor wavelet (Gaussian modulated cosine)"""
    t = torch.linspace(-nt//2, nt//2, nt) * dt
    sigma = 1.0 / (2 * math.pi * f)
    w = torch.exp(-t**2 / (2 * sigma**2)) * torch.cos(2 * math.pi * f * t)
    return w / torch.max(torch.abs(w))

def ormsby_wavelet(f1, f2, f3, f4, dt, nt):
    """Ormsby wavelet (bandpass trapezoidal)"""
    t = torch.linspace(-nt//2, nt//2, nt) * dt
    def sinc(x): return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(math.pi * x) / (math.pi * x))
    w = (
        (f4 * sinc(f4 * t))**2
        - (f3 * sinc(f3 * t))**2
        - (f2 * sinc(f2 * t))**2
        + (f1 * sinc(f1 * t))**2
    ) / ((f4 - f3) + (f2 - f1))
    return w / torch.max(torch.abs(w))

def klauder_wavelet(f1, f2, T, dt, nt=97):
    """Klauder wavelet (chirp-like sweep autocorrelation)"""
    t = torch.linspace(0, T, nt)
    # Linear chirp
    sweep = torch.cos(2 * math.pi * (f1 * t + (f2 - f1) * t**2 / (2 * T)))
    # autocorrelation to create Klauder wavelet
    w = torch.nn.functional.conv1d(
        sweep.view(1,1,-1), sweep.flip(0).view(1,1,-1), padding=len(t)//2
    ).flatten()
    if w.numel() != nt:
        w = torch.nn.functional.interpolate(
            w.view(1,1,-1), size=nt, mode='linear', align_corners=False
        ).flatten()

    w = w / w.abs().max()
    return w

def sinc_wavelet(f, dt, nt):
    """Sinc wavelet (band-limited impulse)"""
    t = torch.linspace(-nt//2, nt//2, nt) * dt
    w = torch.where(t == 0, torch.tensor(1.0, device=t.device), torch.sin(2 * math.pi * f * t) / (2 * math.pi * f * t))
    return w / torch.max(torch.abs(w))


__all__ = ['extract_impedance', 'extract_reflectivity', 'extract_seismic', 'add_awgn', 'generate_twt', 'generate_warped_twt', 'depth2time_interpolation', 'resample_logs_to_seismic', 'ricker_wavelet', 'gabor_wavelet', 'ormsby_wavelet', 'klauder_wavelet', 'sinc_wavelet', 'ricker']