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
    return np.convolve(r, w)

def distort_tdr(tdr, sigma=5, scale=10):
    noise = np.convolve(np.random.randn(len(tdr)),
                         np.exp(-np.linspace(-2, 2, sigma)**2),
                         mode="same")
    noise = noise / np.max(np.abs(noise))
    shift = noise * scale

    tdr_distorted = tdr + shift
    return tdr_distorted, shift

def add_awgn(s, snr_db):
    s_power = np.mean(s**2)

    snr_linear = 10**(snr_db/10)
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
    t = torch.linspace(-nt//2, nt//2, nt) * dt
    pi2 = (math.pi * f) ** 2
    w = (1 - 2 * pi2 * t**2) * torch.exp(-pi2 * t**2)
    return w / torch.max(torch.abs(w))

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


def power_spectrum(a):
    a = torch.tensor(a)
    af = torch.fft.rfft(a, dim=-1)

    ap = torch.abs(af)**2

    ap = torch.clamp(ap, min=1e-8)

    ap = ap / (torch.sum(ap, dim=-1, keepdim=True) + 1e-8)

    return ap

__all__ = ['extract_impedance', 'extract_reflectivity', 'extract_seismic', 'distort_tdr', 'add_awgn', 'generate_distort_tdr', 'ricker_wavelet', 'gabor_wavelet', 'ormsby_wavelet', 'klauder_wavelet', 'sinc_wavelet', 'power_spectrum']