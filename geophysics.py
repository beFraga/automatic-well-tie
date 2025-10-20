import numpy as np
import torch

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

__all__ = ['extract_impedance', 'extract_reflectivity', 'extract_seismic', 'distort_tdr', 'add_awgn', 'generate_distort_tdr']