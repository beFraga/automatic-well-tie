import numpy as np
from scipy.linalg import toeplitz

def AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv):
    """
    AKI RICHARDS COEFFICIENTS MATRIX
    Computes the Aki Richards coefficient matrix.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Vp : array_like 
        P-wave velocity profile (km/s).
    Vs : float or array_like 
        S-wave velocity profile (km/s).
    theta : float or array_like
        Reflection angles.
    nv : int
        Number of model variables.

    Returns
    -------
    A : array_like
        Aki Richards coefficients matrix.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    # initial parameters
    nsamples = Vp.shape[0]
    ntheta = len(theta)
    A = np.zeros(( (nsamples-1)*ntheta, nv*(nsamples-1)))

    # average velocities at the interfaces
    avgVp = 1 / 2 * (Vp[0:- 1] + Vp[1:])
    avgVs = 1 / 2 * (Vs[0:- 1] + Vs[1:])

    # reflection coefficients (Aki Richards linearized approximation)
    for i in range(ntheta):
        cp = 1 / 2 * (1 + np.tan(theta[i]*np.pi / 180) ** 2) * np.ones(nsamples - 1)
        cs = -4 * (avgVs ** 2) / (avgVp ** 2) * np.sin(theta[i]*np.pi / 180) ** 2
        cr = 1 / 2 * (1 - 4 * (avgVs ** 2) / (avgVp ** 2) * np.sin(theta[i]*np.pi / 180) ** 2)
        Acp = np.diag(cp)
        Acs = np.diag(cs)
        Acr = np.diag(cr)
        A[ i*(nsamples-1) : (i+1)*(nsamples-1), : ] = np.hstack([Acp, Acs, Acr])
   
    return A
    
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

def SeismicModel(Vp, Vs, Rho, Time, theta, wavelet):
    """
    SEISMIC MODEL
    Computes synthetic seismic data according to a linearized seismic model
    based on the convolution of a wavelet and the linearized approximation
    of Zoeppritz equations.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    Vp : array_like
        P-wave velocity profile.
    Vs : array_like
        S-wave velocity profile.
    Rho : array_like
        Density profile.
    theta : array_like
        Vector of reflection angles.
    wavelet : array_like
        Wavelet.

    Returns
    -------
    Seis : array_like
        Vector of seismic data (nsamples x nangles, 1).
    Time : array_like
        Seismic times (nsamples, 1).

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    # initial parameters
    ntheta = len(theta)
    nm = Vp.shape[0]

    # number of variables
    nv = 3

    # logarithm of model variables
    logVp = np.log(Vp)
    logVs = np.log(Vs)
    logRho = np.log(Rho)
    m = np.hstack([logVp, logVs, logRho])
    m = m.reshape(len(m),1)

    # Aki Richards matrix
    A = AkiRichardsCoefficientsMatrix(Vp, Vs, theta, nv)

    # Differential matrix 
    D = DifferentialMatrix(nm, nv)
    mder = np.dot(D, m)

    # Reflectivity coefficients matrix
    Cpp = np.dot(A, mder)

    # Wavelet matrix
    W = WaveletMatrix(wavelet, nm, ntheta)

    # Seismic data matrix
    Seis = np.dot(W, Cpp)

    # Time seismic measurements
    TimeSeis = 1 / 2 * (Time[0:- 1] + Time[1:])
    
    return Seis, TimeSeis
    
def WaveletMatrix(wavelet, nsamples, ntheta):
    """
    WAVELET MATRIX
    Computes the wavelet matrix for discrete convolution.
    Written by Dario Grana (August 2020)

    Parameters
    ----------
    w : array_like
        Wavelet.
    ns : int
        Number of samples.
    ntheta : int
        Number of angles.

    Returns
    -------
    W : array_like
        Wavelet matrix.

    References: Grana, Mukerji, Doyen, 2021, Seismic Reservoir Modeling: Wiley - Chapter 5.1
    """

    W = np.zeros((ntheta*(nsamples-1), ntheta*(nsamples-1)))
    indmaxwav = np.argmax(wavelet)
    
    for i in range(ntheta):
        wsub = convmtx(wavelet, (nsamples - 1))
        wsub = wsub.T
        W[ i*(nsamples-1):(i+1)*(nsamples-1), i*(nsamples-1):(i+1)*(nsamples-1)] = wsub[indmaxwav:indmaxwav+(nsamples-1),:]
    
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