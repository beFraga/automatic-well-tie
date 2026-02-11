import numpy as np
import scipy.interpolate
import torch
import math
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz

def extract_impedance(rho, vp):
    return rho * vp

def extract_reflectivity(z):
    zi = z[:-1]
    zii = z[1:]
    r = (zii - zi) / (zii + zi)
    return np.concatenate(([0], r))

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


def NormalIncidenceMatrix(nsamples):
    """
    NORMAL INCIDENCE MATRIX
    Computes the linearized reflection coefficient matrix for normal incidence.
    Replaces AkiRichardsCoefficientsMatrix for zero-offset modeling.
    
    Parameters
    ----------
    nsamples : int
        Number of samples in the profile.

    Returns
    -------
    A : array_like
        Coefficients matrix (nsamples-1, 2*(nsamples-1)).
    """
    
    # Número de interfaces
    n_interfaces = nsamples - 1
    
    # Em incidência normal linearizada, os pesos são 0.5 para Vp e 0.5 para Rho
    # R ~ 0.5 * dlnVp + 0.5 * dlnRho
    weight = 0.5 * np.eye(n_interfaces)
    
    # Empilha as matrizes diagonais horizontalmente: [Peso_Vp, Peso_Rho]
    A = np.hstack([weight, weight])
    
    return A


def SeismicModel(Vp, Rho, wavelet, t0=0, dt=0.002):
    """
    Versão otimizada que não exige um array de tempo de entrada.
    Assume que Vp e Rho já estão amostrados no 'dt' correto.
    """
    
    # 1. Parâmetros iniciais
    nm = Vp.shape[0]     # Número de amostras do modelo (camadas)
    ns_seis = nm - 1     # Número de amostras da sísmica (interfaces)
    
    # 2. Logaritmos e Vetor de Modelo (m)
    # AVISO: Evitar log(0) ou log(negativo) se houver nans
    logVp = np.log(Vp)
    logRho = np.log(Rho)
    
    m = np.hstack([logVp, logRho]).reshape(-1, 1)

    # 3. Matrizes Operacionais
    # Matriz de Incidência Normal (pesos 0.5)
    A = NormalIncidenceMatrix(nm)
    
    # Matriz Diferencial (Derivada discreta) - nv=2
    D = DifferentialMatrix(nm, 2)
    
    # Matriz da Wavelet (Convolução)
    W = WaveletMatrixZeroOffset(wavelet, nm)

    # 4. Cálculo (Forward Model)
    # mder = refletividade (dlnVp, dlnRho)
    mder = np.dot(D, m) 
    
    # Cpp = Coeficiente de Reflexão (R)
    R = np.dot(A, mder)
    
    # Sismica Sintética
    Seis = np.dot(W, R)

    # 5. Criação do Vetor de Tempo de Saída
    # O tempo da primeira reflexão ocorre em t0 + dt/2
    # Ex: Se t0=0 e dt=4ms, a primeira interface (entre amostra 0 e 1) está em 2ms.
    TimeSeis = np.arange(0, ns_seis) * dt + t0 + (dt / 2)
    
    return Seis, TimeSeis

def WaveletMatrixZeroOffset(wavelet, nsamples):
    """
    WAVELET MATRIX ZERO OFFSET
    Computes the wavelet matrix for discrete convolution (single trace).
    """
    # Dimensão reduzida pois não há empilhamento de ângulos
    W = np.zeros((nsamples - 1, nsamples - 1))
    indmaxwav = np.argmax(wavelet)
    
    wsub = convmtx(wavelet, (nsamples - 1))
    wsub = wsub.T
    
    # Preenche a matriz (equivalente a convolução 1D)
    W[:, :] = wsub[indmaxwav:indmaxwav + (nsamples - 1), :]
    
    return W

def convmtx(w, ns):
    """    
    CONVMTX
    Computes the Toeplitz matrix for discrete convolution.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    w : array_like
        Wavelet.
    ns : int
        Numbr of samples.

    Returns
    -------
    C : array_like
        Toeplitz matrix.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    if len(w) < ns:
        a = np.r_[w[0], np.zeros(ns-1)]
        b = np.r_[w, np.zeros(ns-1)]
    else:
        b = np.r_[w[0], np.zeros(ns - 1)]
        a = np.r_[w, np.zeros(ns - 1)]
    C = toeplitz(a, b)

    return C


def DifferentialMatrix(nt, nv):
    """
    DIFFERENTIAL MATRIX
    Computes the differential matrix for discrete differentiation.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    nt : int
        Number of samples.
    nv : int
        Number of model variables.

    Returns
    -------
    D : array_like 
        Differential matrix.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    I = np.eye(nt)
    B = np.zeros((nt, nt))
    B[1:, 0:- 1] = -np.eye(nt-1)
    I = (I + B)
    J = I[1:,:]
    D = np.zeros(((nt-1)*nv, nt*nv))
    for i in range(nv):
        D[ i*(nt-1):(i+1)*(nt-1),i*nt:(i+1)*nt] = J
        
    return D

def generate_twt(sonic, depth, start_twt=0.0):
    sonic = np.asarray(sonic)
    depth = np.asarray(depth)

    print("sonic: ", sonic)
    print("depth: ", depth)

    dx = np.diff(depth)
    sonic_mean = 0.5 * (sonic[:-1] + sonic[1:])
    dt = sonic_mean * dx # sonic[:-1]
    t = np.concatenate(([0], np.cumsum(dt))) * 1e-6
    t = 2 * t + start_twt

    return t

def generate_warped_twt(tdr, max_shift_ms=0.05, smoothness=5):
    n_samples = len(tdr)

    noise = np.random.normal(0, 1, n_samples)
    kernel = np.ones(smoothness) / smoothness
    smooth_noise = np.convolve(noise, kernel, mode='same')

    time_shift = (smooth_noise / np.max(np.abs(smooth_noise))) * max_shift_ms

    tdr_warped = tdr + time_shift

    return tdr_warped, time_shift

def resample_logs_to_seismic(twt, log, dt, method='backus'):
    """
    method : str
        'linear': Interpolação linear simples (rápido, mas pode causar aliasing).
        'block': Média aritmética em blocos (melhor para densidade).
        'backus': (Simplificado) Média harmônica para Vp (fisicamente correto para ondas).
    """

    t_min = np.ceil(twt.min() / dt) * dt
    t_max = np.floor(twt.max() / dt) * dt
    print("min: ", t_min, "max: ", t_max, "dt: ", dt)
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


__all__ = ['extract_impedance', 'extract_reflectivity', 'extract_seismic', 'add_awgn', 'SeismicModel', 'generate_twt', 'generate_warped_twt', 'depth2time_interpolation', 'resample_logs_to_seismic', 'ricker_wavelet', 'gabor_wavelet', 'ormsby_wavelet', 'klauder_wavelet', 'sinc_wavelet']