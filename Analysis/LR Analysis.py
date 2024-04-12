# -*- coding: utf-8 -*-
"""
Locking Region Analysis v1.0

Lukas Kostal, 18.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.optimize as so


def norm(arr):
    arr -= np.amin(arr)
    arr /= np.amax(arr)
    return arr


def lpf(arr, fc, fs, order=2):
    filt = ss.butter(order, fc, fs=fs, btype='low', output='sos')
    arr = ss.sosfilt(filt, arr)
    return arr


def lin(x, m, c):
    y = m * x + c
    return y


path = '/Users/lukaskostal/Desktop/Project/Data/lock region_1.CSV'

fsr = 1.5e9

fi = 0.3e9
ff = 2.2e9

f_abs = 581.9483910e12
f_ref = 0.8e9

data = np.loadtxt(f'{path}', delimiter=',', unpack=True, comments='#', skiprows=1)

t = data[0, :]
t -= t[0]

V_dfs = data[3, :]
I_fpe = data[4, :]
V_rmp = data[2, :]

fs = 1 / np.mean(np.diff(t))
V_rmp = lpf(V_rmp, 10, fs)

I_fpe = norm(I_fpe)

pks_p, _ = ss.find_peaks(V_rmp, prominence=0.1)
pks_n, _ = ss.find_peaks(-V_rmp, prominence=0.1)

idx_n = pks_n[0]
idx_p = pks_p[pks_p > idx_n][0]

plt.plot(t, V_rmp, c='C0')
plt.plot(t[idx_n : idx_p], V_rmp[idx_n : idx_p], c='C1')
plt.plot(t, I_fpe, c='C2')
plt.plot(t[pks_p], V_rmp[pks_p], 'x', c='r')
plt.plot(t[pks_n], V_rmp[pks_n], 'x', c='b')
plt.show()

t = t[idx_n : idx_p]
V_dfs = V_dfs[idx_n : idx_p]
V_rmp = V_rmp[idx_n : idx_p]
I_fpe = I_fpe[idx_n : idx_p]

pks, _ = ss.find_peaks(I_fpe, height=0.9, prominence=0.5, width=4)

if len(pks) > 2:
    raise Exception("more than 2 FPE transmission peaks detected")
    
popt, pcov = so.curve_fit(lin, t, V_rmp)

cal_tf = fsr / np.diff(t[pks])

f = t * cal_tf

cal_vf = fsr / np.diff(lin(t[pks], *popt))

print(cal_vf[0])

plt.plot(t, V_rmp, c='C0')
plt.plot(t, I_fpe, c='C2')
plt.plot(t[pks], I_fpe[pks], 'x', c='r')
plt.show()

idx_i = np.argmin(np.abs(f - fi))
idx_f = np.argmin(np.abs(f - ff))

f -= f_ref

f = f[idx_i : idx_f]
V_dfs = V_dfs[idx_i : idx_f]

# parameters for plotting
plt.figure(1, figsize=(10, 4))
plt.title('Locking Region Spectrum')
    
plt.xlabel(r'scan frequency $\Delta f$ (GHz)')
plt.ylabel(r'lock-in signal (V)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()


plt.axvline(x=0, ls='--', c='r')
plt.plot(f*1e-9, V_dfs)

plt.figtext(0.63, 0.19, f'$f = \Delta f + ${f_abs*1e-12:.6f} THz', \
            bbox=dict(facecolor='none', pad=0.3, boxstyle='round'))

plt.savefig(f'Output/DFS lock region.png', dpi=300, bbox_inches='tight')
plt.show()
