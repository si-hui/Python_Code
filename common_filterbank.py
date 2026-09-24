import numpy as np
from scipy.signal import remez, firwin

def opt_filter(filtorder: int, N: int):
    """
    Python translation of opt_filter.m
    Designs the optimal lowpass prototype filter for Pseudo-QMF banks
    using the Parks-McClellan (remez) algorithm.
    """
    stopedge = 1.0 / N       # Stopband edge fixed at (1/N)*pi
    passedge = 1.0 / (4 * N) # Initial passband edge
    tol = 1e-6
    step = 0.1 * passedge
    way = -1
    pcost = 10.0
    flag = False

    while not flag:
        # Remez exchange algorithm (fs=2.0 normalizes Nyquist to 1.0)
        p = remez(filtorder + 1, [0, passedge, stopedge, 1.0], [1, 0], weight=[5, 1], fs=2.0)
        
        P = np.fft.fft(p, 4096)
        OptRange = int(np.floor(2048 / N))
        phi = np.zeros(OptRange)
        
        for k in range(1, OptRange + 1):
            phi[k-1] = np.abs(P[OptRange - k + 1])**2 + np.abs(P[k - 1])**2
            
        tcost = np.max(np.abs(phi - 1.0))
        
        if tcost > pcost:
            step /= 2.0
            way = -way
            
        if np.abs(pcost - tcost) < tol:
            flag = True
            
        pcost = tcost
        passedge += way * step
        
    return p, passedge


def make_bank(p: np.ndarray, N: int):
    """
    Python translation of make_bank.m (Cosine Modulation, Eq 2.35)
    """
    L = len(p)
    H = np.zeros((N, L))
    n = np.arange(L)
    
    for i in range(N):
        H[i, :] = 2 * p * np.cos((np.pi / N) * (i + 0.5) * (n - (L - 1) / 2.0) + ((-1)**i) * (np.pi / 4.0))
        
    F = np.fliplr(H)
    return H, F


def make_bank_DFT(p: np.ndarray, N: int):
    """
    Python translation of make_bank_DFT.m
    """
    flen = len(p)
    H = np.zeros((flen, N // 2 + 1), dtype=complex)
    n = np.arange(flen)
    
    H[:, 0] = p
    H[:, N // 2] = p * ((-1)**n)
    
    for k in range(1, N // 2):
        H[:, k] = p * np.exp(1j * 2 * np.pi / N * k * n)
        
    D = H[:, 1 : N // 2]
    H_full = np.hstack((H, np.conj(np.fliplr(D))))
    F_full = H_full
    return H_full.T, F_full.T # Shape (N, L)


def FreqResp(H: np.ndarray, DFTpoint: int = 4096):
    """
    Computes Frequency Response over [0, 2*pi]
    """
    N, L = H.shape
    Hz = np.zeros((N, DFTpoint), dtype=complex)
    w = np.linspace(0, 2 * np.pi, DFTpoint, endpoint=False)
    for k in range(N):
        Hz[k, :] = np.fft.fft(H[k, :], DFTpoint)
    return Hz, w