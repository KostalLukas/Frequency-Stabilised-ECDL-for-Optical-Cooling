# -*- coding: utf-8 -*-
"""
Doppler Broadened Spectrum Analysis v1.0

Lukas Kostal, 18.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.interpolate as si


def norm(arr):
    arr -= np.amin(arr)
    arr /= np.amax(arr)
    return arr


def hpf(arr, fc, fs, order=2):
    filt = ss.butter(order, fc, fs=fs, btype='high', output='sos')
    arr = ss.sosfilt(filt, arr)
    return arr

def lpf(arr, fc, fs, order=2):
    filt = ss.butter(order, fc, fs=fs, btype='low', output='sos')
    arr = ss.sosfilt(filt, arr)
    return arr


path = '/Users/lukaskostal/Desktop/Project/Data/large scan_1.CSV'

fsr = 1.5e9

f_abs = 581.9483910e12
f_ref = 12.860e9

data = np.loadtxt(f'{path}', delimiter=',', unpack=True, comments='#', skiprows=1)

t = data[0, :]
t -= t[0]

I_dbs = data[1, :]
V_rmp = data[2, :]
I_dfs = data[3, :]
I_fpe = data[4, :]

I_dbs = norm(I_dbs)
I_dfs = norm(I_dfs)
I_fpe = norm(I_fpe)

fs = 1 / np.mean(np.diff(t))

I_dfs = hpf(I_dfs, 1, fs)
I_dfs = lpf(I_dfs, 40, fs)

pks_p, _ = ss.find_peaks(V_rmp, prominence=1)
pks_n, _ = ss.find_peaks(-V_rmp, prominence=1)

plt.plot(t, V_rmp)
plt.plot(t[pks_p], V_rmp[pks_p], 'x')
plt.plot(t[pks_n], V_rmp[pks_n], 'x')
plt.show()

idx_n = pks_n[0]
idx_p = pks_p[pks_p > idx_n][0]

t = t[idx_n : idx_p]
I_dbs = I_dbs[idx_n : idx_p]
I_dfs = I_dfs[idx_n : idx_p]
I_fpe = I_fpe[idx_n : idx_p]

pks, _ = ss.find_peaks(I_fpe, height=0.3, prominence=0.1, width=10)

f_cal = np.arange(0, len(pks)) * fsr
t_cal = t[pks]

fcal = si.interp1d(t_cal, f_cal, fill_value='extrapolate')

f = fcal(t)

f -= f_ref

plt.plot(t, I_fpe)
plt.plot(t[pks], I_fpe[pks], 'x')
plt.show()

# parameters for plotting
plt.figure(1, figsize=(8, 4))
plt.title('Iodine Doppler Broadened Spectrum')
    
plt.xlabel(r'scan frequency $\Delta f$ (GHz)')
plt.ylabel(r'normalised intensity $I$ (arb.u.)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f*1e-9, I_dbs, c='C0')

plt.figtext(0.63, 0.19, f'$f = \Delta f + ${f_abs*1e-12:.6f} THz', \
            bbox=dict(facecolor='none', pad=0.3, boxstyle='round'))

plt.show()

# parameters for plotting
plt.figure(1, figsize=(8, 5))
plt.title('Iodine Doppler Broadened Spectrum')
    
plt.xlabel(r'scan frequency $\Delta f$ (GHz)')
plt.ylabel(r'lock-in signal (V)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f*1e-9, I_dfs, c='C0')

plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(11, 5))

ax1.set_title('Measured Molecular Iodine Spectrum')

ax1.axvline(x=0, ls='--', c='r')
ax2.axvline(x=0, ls='--', c='r')

ax1.plot(f*1e-9, I_dbs, c='C2', label='Doppler broadened')
ax2.plot(f*1e-9, I_dfs, c='C0', label='sub-Doppler')

ax1.legend(loc=4)
ax2.legend(loc=4)

ax2.set_xlabel(r'scan frequency $\Delta f$ (GHz)')
ax1.set_ylabel(r'normalised intensity (arb.u.)')
ax2.set_ylabel(r'lock-in signal (V)')

plt.figtext(0.70, 0.04, f'$f = \Delta f + ${f_abs*1e-12:.6f} THz', \
            bbox=dict(facecolor='none', pad=0.3, boxstyle='round'))    

plt.rc('grid', linestyle=':', color='black', alpha=0.8)
ax1.grid()
ax2.grid()
    
fig.subplots_adjust(hspace=0.06)

plt.savefig(f'Output/DFS Iodine spectrum.png', dpi=300, bbox_inches='tight')
plt.show()