import numpy as np
import numpy.linalg as lin
import numpy.random as npr 
import scipy.optimize as sco 
import time
from scipy import interpolate 
from scipy.linalg import expm
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

S0 = 100.0 # ini stock price
G = 100.0 # garantee level
T = 10 # maturity
m = 0.02 # R&E fee
me = 0.01441466502048007 # rider charge
md = 0.0008662755495145845
N, X = 1024, 7.5 # FST param.
I, Path = 100000, 1000 # Monte Carlo param.
r = [0.04, 0.01] # risk-free interest rate
drift = [0.05, 0.02] # drift
sigma = [0.1, 0.2] # volatility
A = np.array([[-0.4, 0.4], [0.3, -0.3]]) # transition matrix
Pi = np.array([[0.9999], [0.0001]]) # ini state

def z(r, T, A, Pi):
    r1, r2 = r[0], r[1]
    dr = r1 - r2
    D = np.array([[1, 0], [0, 0]])
    E = expm((A - dr * D) * T)
    One = np.ones((2, 1))
    Z = np.exp(-r2 * T) * Pi.T.dot(E.dot(One))
    return float(Z)
# Rho1 of Zero-coupon bond
def rho1_z(r, T, A, Pi):
    r1, r2 = r[0], r[1]
    dr = r1 - r2
    D = np.array([[1, 0], [0, 0]])
    E = expm((A - dr * D) * T)
    One = np.ones((2, 1))
    Z = -np.exp(-r2 * T) * Pi.T.dot(T * D.dot(E.dot(One))) / 100
    return float(Z)
# Rho2 of Zero-coupon bond
def rho2_z(r, T, A, Pi):
    r1, r2 = r[0], r[1]
    dr = r1 - r2
    D = np.array([[1, 0], [0, 0]])
    E = expm((A - dr * D) * T)
    One = np.ones((2, 1))
    Z1 = np.exp(-r2 * T) * Pi.T.dot(T * D.dot(E.dot(One)))
    Z2 = -T * np.exp(-r2 * T) * Pi.T.dot(E.dot(One))
    Z = (Z1 + Z2) / 100
    return float(Z)
print('z(10) = %.16f' % z(r, T, A, Pi)) # 0.7784180374597731
print('rho1_z(10) = %.16f' % rho1_z(r, T, A, Pi)) # -0.0778336802081909
print('rho2_z(10) = %.16f' % rho2_z(r, T, A, Pi)) # -0.0000081235377864

# %%
# Stochastic Motality Risk
def b(t):
    a, sig = 0.08000149236993707, 0.010468669959318862
    b = -np.sqrt(a ** 2 + 2 * sig ** 2)
    c = (b + a) / 2
    d = (b - a) / 2
    B = (1 - np.exp(b * t)) / (c + d * np.exp(b * t))
    return B
def s(t):
    ini = 0.00495 # aged 50 years old in 2017
    # ini = 0.01164 # aged 60 years old in 2017
    # ini = 0.02255 # aged 70 years old in 2017
    a, sig = 0.08000149236993707, 0.010468669959318862
    b = -np.sqrt(a ** 2 + 2 * sig ** 2)
    c = (b + a) / 2
    d = (b - a) / 2
    # A = 0
    # S_feller = np.exp(A + B(t)*ini)
    S = np.exp((1 - np.exp(b * t)) / (c + d * np.exp(b * t)) * ini)
    return S
print('s(0) = %.16f' % s(0)) # 1.0
print('s(10) = %.16f' % s(15)) # 0.8672053048004046

# %%
def rsln_put_fst(S0, G, T, r, sigma, m, A, Pi, N, X):
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    # Payoff function at time T
    ST = S0 * np.exp(x)
    payoff_p = np.maximum(G - ST, 0)
    # Matrix characteristic funciton
    psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
    psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
    Psi = np.zeros((N, 2, 2), dtype=complex)
    for i in range(N):
        Psi[i, 0] = [A11 + psi1[i], A12]
        Psi[i, 1] = [A21, A22 + psi2[i]]
    char = np.array([expm(i * T) for i in Psi])
    # FST method
    p1 = ifft((char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_p)).real
    p2 = ifft((char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_p)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, p1)
    f2 = interpolate.PchipInterpolator(ST, p2)
    P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]]))
    return float(P) # 13.8429656013946278
print('Put_FST = %.16f' % rsln_put_fst(S0, G, T, r, sigma, m, A, Pi, N, X))

# %%
def rsln_gmmb_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    P = rsln_put_fst(S0, G, T, r, sigma, m, A, Pi, N, X)
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method
        re1 = ifft((char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_r)).real
        re2 = ifft((char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, re1)
        f2 = interpolate.PchipInterpolator(ST, re2)
        Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    Re = np.sum(Re)
    # calculate GMMB loss
    Loss = s(T) * P - Re
    return float(Loss)  # 0.0000000000005329
    
start = time.clock()
print('GMMB_FST = %.16f' % rsln_gmmb_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X))
elapsed = (time.clock() - start)
print("Time used:", elapsed)

# %%
def rsln_putdelta_fst(S0, G, T, r, sigma, m, A, Pi, N, X):
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    # Payoff function at time T
    ST = S0 * np.exp(x)
    payoff_p = np.maximum(G - ST, 0)
    # Matrix characteristic funciton
    psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
    psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
    Psi = np.zeros((N, 2, 2), dtype=complex)
    for i in range(N):
        Psi[i, 0] = [A11 + psi1[i], A12]
        Psi[i, 1] = [A21, A22 + psi2[i]]
    char = np.array([expm(i * T) for i in Psi])
    # FST method
    delta_p1 = (ifft(1j * w * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_p)) / (ST)).real
    delta_p2 = (ifft(1j * w * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_p)) / (ST)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, delta_p1)
    f2 = interpolate.PchipInterpolator(ST, delta_p2)
    Delta_P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]]))
    return float(Delta_P) # -0.2892064937563866
print('Delta_Put = %.16f' % rsln_putdelta_fst(S0, G, T, r, sigma, m, A, Pi, N, X))

# %%
def rsln_gmmbdelta_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    Delta_P = rsln_putdelta_fst(S0, G, T, r, sigma, m, A, Pi, N, X)
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Delta_Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method
        delta_re1 = (ifft(1j * w * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_r)) / (ST)).real
        delta_re2 = (ifft(1j * w * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_r)) / (ST)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, delta_re1)
        f2 = interpolate.PchipInterpolator(ST, delta_re2)
        Delta_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    Delta = s(T) * Delta_P - np.sum(Delta_Re)
    return float(Delta) # -0.4007022754308916
print('Delta_GMMB = %.16f' % rsln_gmmbdelta_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X))

# %%
def rsln_putgamma_fst(S0, G, T, r, sigma, m, A, Pi, N, X):
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    # Payoff function at time T
    ST = S0 * np.exp(x)
    payoff_p = np.maximum(G - ST, 0)
    # Matrix characteristic funciton
    psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
    psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
    Psi = np.zeros((N, 2, 2), dtype=complex)
    for i in range(N):
        Psi[i, 0] = [A11 + psi1[i], A12]
        Psi[i, 1] = [A21, A22 + psi2[i]]
    char = np.array([expm(i * T) for i in Psi])
    # FST method
    gamma_p1 = (ifft(-(1j * w + w ** 2) * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_p)) / (ST ** 2)).real
    gamma_p2 = (ifft(-(1j * w + w ** 2) * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_p)) / (ST ** 2)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, gamma_p1)
    f2 = interpolate.PchipInterpolator(ST, gamma_p2)
    Gamma_P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]]))
    return float(Gamma_P) # 0.00622554 54081478
print('Gamma_Put = %.16f' % rsln_putgamma_fst(S0, G, T, r, sigma, m, A, Pi, N, X))

# %%
def rsln_gmmbgamma_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    Gamma_P = rsln_putgamma_fst(S0, G, T, r, sigma, m, A, Pi, N, X)
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Gamma_Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method
        gamma_re1 = (ifft(-(1j * w + w ** 2) * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_r)) / (ST ** 2)).real
        gamma_re2 = (ifft(-(1j * w + w ** 2) * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_r)) / (ST ** 2)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, gamma_re1)
        f2 = interpolate.PchipInterpolator(ST, gamma_re2)
        Gamma_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    # calculate GMMB delta
    Gamma = s(T) * Gamma_P - np.sum(Gamma_Re)
    return float(Gamma) # 0.0060459769788203
print('Gamma_GMMB = %.16f' % rsln_gmmbgamma_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X))

# %%
def rsln_putvega_fst(S0, G, T, r, sigma, m, A, Pi, N, X):
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    # Payoff function at time T
    ST = S0 * np.exp(x)
    payoff_p = np.maximum(G - ST, 0)
    # Matrix characteristic funciton
    psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
    psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
    Psi = np.zeros((N, 2, 2), dtype=complex)
    for i in range(N):
        Psi[i, 0] = [A11 + psi1[i], A12]
        Psi[i, 1] = [A21, A22 + psi2[i]]
    char = np.array([expm(i * T) for i in Psi])
    # FST method for sigma 1
    vega1_p1 = ifft(-(1j * w + w ** 2) * sigma1 * T * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_p)).real
    vega1_p2 = ifft(0 * fft(payoff_p)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, vega1_p1)
    f2 = interpolate.PchipInterpolator(ST, vega1_p2)
    Vega1_P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])) / 100
    # FST method for sigma 2
    vega2_p1 = ifft(0 * fft(payoff_p)).real
    vega2_p2 = ifft(-(1j * w + w ** 2) * sigma2 * T * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_p)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, vega2_p1)
    f2 = interpolate.PchipInterpolator(ST, vega2_p2)
    Vega2_P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])) / 100
    return (float(Vega1_P), float(Vega2_P))
Results = rsln_putvega_fst(S0, G, T, r, sigma, m, A, Pi, N, X) 
print('Vega1_Put = %.16f' % Results[0]) # 0.6225024163399533
print('Vega2_Put = %.16f' % Results[1]) # 0.0001175009773657

# %%
def rsln_gmmbvega_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    Vega1_P = rsln_putvega_fst(S0, G, T, r, sigma, m, A, Pi, N, X)[0]
    Vega2_P = rsln_putvega_fst(S0, G, T, r, sigma, m, A, Pi, N, X)[1]
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Vega1_Re = []
    Vega2_Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method for sigma1
        vega1_re1 = ifft(-(1j * w + w ** 2) * sigma1 * t * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_r)).real
        vega1_re2 = ifft(0 * fft(payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, vega1_re1)
        f2 = interpolate.PchipInterpolator(ST, vega1_re2)
        Vega1_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))           # FST method for sigma2
        vega2_re1 = ifft(0 * fft(payoff_r)).real
        vega2_re2 = ifft(-(1j * w + w ** 2) * sigma2 * t * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, vega2_re1)
        f2 = interpolate.PchipInterpolator(ST, vega2_re2)
        Vega2_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    Vega1_Re = np.sum(Vega1_Re) / 100
    Vega2_Re = np.sum(Vega2_Re) / 100
    # calculate GMMB delta
    Vega1 = (s(T) * Vega1_P - Vega1_Re)
    Vega2 = (s(T) * Vega2_P - Vega2_Re)
    return (float(Vega1), float(Vega2))
Results = rsln_gmmbvega_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X)
print('Vega1_GMMB = %.16f' % Results[0]) # 0.5771623784058360
print('Vega2_GMMB = %.16f' % Results[1]) # 0.0001089427796280

# %%
Pi = np.array([[0.9999], [0.0001]]) # ini state
def rsln_putrho_fst(S0, G, T, r, sigma, m, A, Pi, N, X):
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    # Payoff function at time T
    ST = S0 * np.exp(x)
    payoff_p = np.maximum(G - ST, 0)
    # Matrix characteristic funciton
    psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
    psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
    Psi = np.zeros((N, 2, 2), dtype=complex)
    for i in range(N):
        Psi[i, 0] = [A11 + psi1[i], A12]
        Psi[i, 1] = [A21, A22 + psi2[i]]
    char = np.array([expm(i * T) for i in Psi])
    # FST method for rho 1
    rho1_p1 = ifft((1j * w - 1) * T * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_p)).real
    rho1_p2 = ifft(0 * fft(payoff_p)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, rho1_p1)
    f2 = interpolate.PchipInterpolator(ST, rho1_p2)
    Rho1_P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])) / 100
    # FST method for rho 2
    rho2_p1 = ifft(0 * fft(payoff_p)).real
    rho2_p2 = ifft((1j * w - 1) * T * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_p)).real
    # Interpolate prices
    f1 = interpolate.PchipInterpolator(ST, rho2_p1)
    f2 = interpolate.PchipInterpolator(ST, rho2_p2)
    Rho2_P = Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])) / 100
    return (float(Rho1_P), float(Rho2_P))
Results = rsln_putrho_fst(S0, G, T, r, sigma, m, A, Pi, N, X)
print('Rho1_Put = %.16f' % Results[0]) # -4.2758786616469537
print('Rho2_Put = %.16f' % Results[1]) # -0.0004827956080686

# %%
def rsln_gmmbrho_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    Rho1_P = rsln_putrho_fst(S0, G, T, r, sigma, m, A, Pi, N, X)[0]
    Rho2_P = rsln_putrho_fst(S0, G, T, r, sigma, m, A, Pi, N, X)[1]
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Rho1_Re = []
    Rho2_Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method for sigma1
        rho1_re1 = ifft((1j * w - 1) * t * (char[:, 0, 0] + char[:, 0, 1]) * fft(payoff_r)).real
        rho1_re2 = ifft(0 * fft(payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, rho1_re1)
        f2 = interpolate.PchipInterpolator(ST, rho1_re2)
        Rho1_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
        # FST method for sigma2
        rho2_re1 = ifft(0 * fft(payoff_r)).real
        rho2_re2 = ifft((1j * w - 1) * t * (char[:, 1, 0] + char[:, 1, 1]) * fft(payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, rho2_re1)
        f2 = interpolate.PchipInterpolator(ST, rho2_re2)
        Rho2_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    Rho1_Re = np.sum(Rho1_Re) / 100
    Rho2_Re = np.sum(Rho2_Re) / 100
    # calculate GMMB delta
    Rho1 = (s(T) * Rho1_P - Rho1_Re)
    Rho2 = (s(T) * Rho2_P - Rho2_Re)
    return (float(Rho1), float(Rho2))
Results = rsln_gmmbrho_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X)
print('Rho1_GMMB = %.16f' % Results[0]) # -3.9644445280816925
print('Rho2_GMMB = %.16f' % Results[1]) # -0.0004476311322287

# %%
def rsln_gmmbmu_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    P = rsln_put_fst(S0, G, T, r, sigma, m, A, Pi, N, X)
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Mu_Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method
        mu_re1 = ifft((char[:, 0, 0] + char[:, 0, 1]) * fft(b(t) * payoff_r)).real
        mu_re2 = ifft((char[:, 1, 0] + char[:, 1, 1]) * fft(b(t) * payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, mu_re1)
        f2 = interpolate.PchipInterpolator(ST, mu_re2)
        Mu_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    # calculate GMMB mu
    Mu = b(T) * s(T) * P - np.sum(Mu_Re)
    return float(Mu) # -124.7005713467245016
print('Mu_GMMB = %.16f' % rsln_gmmbmu_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X))

# %%
def rsln_gmmbmu2_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X):
    P = rsln_put_fst(S0, G, T, r, sigma, m, A, Pi, N, X)
    # parameter
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    # Real space
    x_min, x_max = -X, X
    dx = (x_max - x_min) / (N - 1)
    x = np.linspace(x_min, x_max, N)
    # Fourier space
    epsilon = 0.0001
    w_max = np.pi / dx
    dw = 2 * w_max / N
    w = np.hstack((np.arange(0, w_max + epsilon, dw), np.arange(-w_max + dw, -dw + epsilon, dw)))
    ST = S0 * np.exp(x)
    # calculate rider charges
    Mu2_Re = []
    for t in range(T):
        # Payoff function at time T
        payoff_r = me * ST * s(t)
        # Matrix characteristic funciton
        psi1 = 1j * (r1 - m - 0.5 * sigma1 ** 2) * w - 0.5 * (sigma1 * w) ** 2 - r1
        psi2 = 1j * (r2 - m - 0.5 * sigma2 ** 2) * w - 0.5 * (sigma2 * w) ** 2 - r2
        Psi = np.zeros((N, 2, 2), dtype=complex)
        for i in range(N):
            Psi[i, 0] = [A11 + psi1[i], A12]
            Psi[i, 1] = [A21, A22 + psi2[i]]
        char = np.array([expm(i * t) for i in Psi])
        # FST method
        mu2_re1 = ifft((char[:, 0, 0] + char[:, 0, 1]) * fft(b(t) ** 2 * payoff_r)).real
        mu2_re2 = ifft((char[:, 1, 0] + char[:, 1, 1]) * fft(b(t) ** 2 * payoff_r)).real
        # Interpolate prices
        f1 = interpolate.PchipInterpolator(ST, mu2_re1)
        f2 = interpolate.PchipInterpolator(ST, mu2_re2)
        Mu2_Re.append(Pi.T.dot(np.array([[f1(S0)], [f2(S0)]])))
    # calculate GMMB mu
    Mu2 = b(T) ** 2 * s(T) * P - np.sum(Mu2_Re)
    return float(Mu2) # 2377.1487055812417566
print('Mu2_GMMB = %.16f' % rsln_gmmbmu2_fst(S0, G, T, r, sigma, m, me, A, Pi, N, X))

# Simulation
def rsln_mc(S0, G, T, r, sigma, m, A, Pi, I):
    r1, r2 = r[0], r[1]
    sigma1, sigma2 = sigma[0], sigma[1]
    l1, l2 = A[0, 1], A[1, 0]
    if Pi[0] > Pi[1]:
        Ini_State = 1.0
    else:
        Ini_State = 2.0
    Stock = np.zeros((T + 1, I))
    for t in range(T + 1):
        for i in range(I):
            LogStock0 = float(np.log(S0))
            Cur_Time = 0.0
            Cur_State = Ini_State
            tau1 = 0.0
            while Cur_Time < t: # Pr(tau_i > t) = exp(-lambda_i * t)
                p = npr.uniform(0, 1)
                if Cur_State == 1:
                    ExpRV = -1 * np.log(p) / l1
                else:
                    ExpRV = -1 * np.log(p) / l2
                if Cur_Time + ExpRV < t and Cur_State == 1:
                    tau1 = tau1 + ExpRV
                else:
                    if Cur_State == 1:
                        tau1 = tau1 + t - Cur_Time
                Cur_Time = Cur_Time + ExpRV
                if Cur_State == 1:
                    Cur_State = 2
                else:
                    Cur_State = 1
            SimRand = float(npr.standard_normal())
            LogStock = LogStock0 + (r1 - m - 0.5 * sigma1 ** 2) * tau1 + (r2 - m - 0.5 * sigma2 ** 2) * (t - tau1) + SimRand * np.sqrt(tau1 * sigma1 ** 2 + (t - tau1) * sigma2 ** 2)
            Stock[t, i] = np.exp(LogStock)   
    return Stock
Stock = rsln_mc(S0, G, T, r, sigma, m, A, Pi, Path)

for j in range(T+1):
    Unhedged = []
    for i in Stock[j]:
        temp = rsln_gmmb_fst(i, G, T-j, r, sigma, m, me, A, Pi, N, X)
        Unhedged.append(temp)
    print(j, np.mean(Unhedged), np.std(Unhedged))
    aa = np.percentile(Unhedged, 5)
    print('VaR_unhedged=', aa)
    bb = [cc for cc in Unhedged if cc < aa]
    print('CVaR_unhedged=', np.mean(bb))

def pv(S, t): # Present Value
    vector = np.zeros(8)
    vector[0] = rsln_gmmb_fst(S, G, t, r, sigma, m, me, A, Pi, N, X)
    for j in range(3): # Put(10), Put(11), Put(12)
        vector[j+1] = rsln_put_fst(S, G, t+j, r, sigma, 0, A, Pi, N, X)
    for j in range(2): # f(15), f(16), Z(15), Z(16)
        vector[j+4] = s(10+j) * z(r, t+j, A, Pi)
        vector[j+6] = z(r, t+j, A, Pi)
    return vector

W_lin = np.array([-1, -228.80152969, 496.9425498, -267.22352691, -7.31831572, 16.75095472, -953.58168627, 948.05690719])
W_reg = np.array([-1, -0.548237040, 0, 0, -3.18881327, 13.5213786, 0, 0])
W_opt = np.array([-1, -2.56736114, 0.07661083, 3.58882855, -7.31825074, 16.75090425, -11.32428309, -10.4227717])

for t in range(T+1): # j = 0,1,2,...,15
    Static = []
    for i in Stock[t]:
        pvt = pv(i, T-t) # present value at time t
        Static.append(pvt.dot(W_lin))
    print(t, np.mean(Static), np.std(Static))
    aa = np.percentile(Static, 5)
    print('VaR_lin=', aa)
    bb = [cc for cc in Static if cc < aa]
    print('CVaR_lin=', np.mean(bb))

for t in range(T+1): # j = 0,1,2,...,15
    Static = []
    for i in Stock[t]:
        pvt = pv(i, T-t) # present value at time t
        Static.append(pvt.dot(W_reg))
    print(t, np.mean(Static), np.std(Static))
    aa = np.percentile(Static, 5)
    print('VaR_reg=', aa)
    bb = [cc for cc in Static if cc < aa]
    print('CVaR_reg=', np.mean(bb))

for t in range(T+1): # j = 0,1,2,...,15
    Static = []
    for i in Stock[t]:
        pvt = pv(i, T-t) # present value at time t
        Static.append(pvt.dot(W_opt))
    print(t, np.mean(Static), np.std(Static))
    aa = np.percentile(Static, 5)
    print('VaR_opt=', aa)
    bb = [cc for cc in Static if cc < aa]
    print('CVaR_opt=', np.mean(bb))