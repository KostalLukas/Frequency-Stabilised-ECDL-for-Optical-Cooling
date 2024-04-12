# -*- coding: utf-8 -*-
"""
Step Response Analysis v1.0

Lukas Kostal, 9.4.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sf


def avg(arr, n):
    arr = arr[:-(len(arr) % n)]
    arr = np.mean(arr.reshape(-1, n), axis=1)
    return arr


def arg(arr):
    phi = np.arctan2(np.imag(arr), np.real(arr))
    
    phi  = phi % (2 * np.pi)
    
    return phi


path = "/Users/lukaskostal/Desktop/Project/Data/step response_2.CSV"

ds = np.loadtxt(path, unpack=True, delimiter=',', skiprows=2)

ds[0] -= ds[0,0]

ti = 0
tf = ds[0, -1]

idx_i = np.argmin(np.abs(ds[0] - ti))
idx_f = np.argmin(np.abs(ds[0] - tf))

ds = ds[:, idx_i : idx_f]

t = ds[0, :]
V1 = ds[1,  :]
V2 = ds[2, :]
V3 = ds[3, :]
V4 = ds[4, :]

t -= t[0]

plt.plot(t, V1)
plt.plot(t, V2)
plt.plot(t, V3)
plt.plot(t, V4)
plt.show()

fs = np.mean(np.diff(t))

f = sf.rfftfreq(len(t), fs)

Vi_fft = sf.rfft(V4)

Vr_fft = sf.rfft(V2)

G = np.abs(Vr_fft / Vi_fft)**2

phi = arg(Vr_fft / Vi_fft) - np.pi

f = avg(f, 100)
G = avg(G, 100)
phi = avg(phi, 100)

plt.plot(f, G)

plt.xscale('log')
plt.yscale('log')

plt.show()

plt.plot(f, phi)

plt.xscale('log')

plt.show()