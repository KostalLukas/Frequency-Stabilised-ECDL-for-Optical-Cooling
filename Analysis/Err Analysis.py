# -*- coding: utf-8 -*-
"""
Oscillating Error Signal Analysis v1.0

Lukas Kostal, 18.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.fft as sf


def rms(arr):
    rms = np.sqrt(np.sum(arr**2) / len(arr))
    return rms

path = '/Users/lukaskostal/Desktop/Project/Data/error oscillation.CSV'

cal_vf = 2031089517.2731104

data = np.loadtxt(f'{path}', delimiter=',', unpack=True, comments='#', skiprows=1)

t = data[0, :]
t -= t[0]

V = data[3, :]

f_std = np.std(V) * cal_vf

fs = 1 / np.mean(np.diff(t))

f, Vfft = ss.periodogram(V, fs, 'flattop', scaling='spectrum')

f_res = f[np.argmax(Vfft)]

print(f'resonant frequency = {f_res*1e-3:.3f} kHz')

# parameters for plotting
plt.figure(1, figsize=(7, 4))
plt.title('Oscillating Error Signal Power Spectrum')
    
plt.xlabel(r'frequency $f$ (Hz)')
plt.ylabel(r'error signal power ($\rm{V}^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f, Vfft)
plt.xscale('log')
plt.yscale('log')

plt.savefig(f'Output/Error oscillation PS.png', dpi=300, bbox_inches='tight')
plt.show()