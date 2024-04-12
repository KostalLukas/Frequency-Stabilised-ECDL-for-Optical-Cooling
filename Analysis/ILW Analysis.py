# -*- coding: utf-8 -*-
"""
Effective Integrated Linewidth Analysis v1.0

Lukas Kostal, 8.4.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.optimize as so
import scipy.fft as sf
import scipy.integrate as sg
import scipy.interpolate as si


def res(x, a0, a1, fsr):
    y = a0 / (1 + a1 * (np.sin(np.pi * x / fsr))**2)
    return y


def inv(x, a0, a1, fsr):
    y = fsr / np.pi * np.sqrt((a0 / x - 1) / a1)
    return y

# function to get indices of rising edge above some threshold
def edge(arr, th):
    sgn = arr >= th
    idx = np.argwhere(np.convolve(sgn, [1, -1]) == 1)
    return idx


fsr = 1.5e9

path_cal = "/Users/lukaskostal/Desktop/Project/Data/calib noise spectrum_2.CSV"
path_mes = "/Users/lukaskostal/Desktop/Project/Data/lock noise spectrum_3.CSV"

ds_cal = np.loadtxt(path_cal, unpack=True, delimiter=',', skiprows=2)
ds_mes = np.loadtxt(path_mes, unpack=True, delimiter=',', skiprows=2)

ch_n = 4

t = ds_cal[0, :]
V_fpe = ds_cal[ch_n, :]
t -= t[0]

heig = 0.8 * np.amax(V_fpe)
pks, _ = ss.find_peaks(V_fpe, height=heig, prominence=0.012, width=4)

plt.plot(t, V_fpe, c='C0')
plt.plot(t[pks], V_fpe[pks], 'x')
plt.show()

cal_tf = fsr / (t[pks[3]] - t[pks[2]])

f = cal_tf * t

t_rng = 1.6
idx = int(t_rng / np.mean(np.diff(t)))

f_fit = f[pks[2]-idx : pks[2]+idx]
V_fit = V_fpe[pks[2]-idx : pks[2]+idx]

f_fit -= f_fit[np.argmax(V_fit)]

fit = lambda x, a0, a1 : res(x, a0, a1, fsr)

ig = np.array([np.amax(V_fit), 1e1])

popt, pcov = so.curve_fit(fit, f_fit, V_fit, p0=ig)

plt.plot(f_fit, V_fit, c='C1')
plt.plot(f_fit, fit(f_fit, *popt))
plt.show()

t = ds_mes[0, :]
V_fpe = ds_mes[ch_n, :]
t -= t[0]

fs = 1 / np.mean(np.diff(t))

N = inv(V_fpe, popt[0], popt[1], fsr)
N[np.isnan(N)] = 0

f, PS = ss.periodogram(N, fs, 'flattop', scaling='spectrum')

# parameters for plotting
plt.figure(1)
plt.title('Frequency Noise Power Spectrum')
    
plt.xlabel(r'reference frequency $f$ (Hz)')
plt.ylabel(r'frequency noise power ($Hz^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f, PS)
plt.xscale('log')
plt.yscale('log')

plt.savefig(f'Output/NFS PS lock.png', dpi=300, bbox_inches='tight')
plt.show()

f, PSD = ss.periodogram(N, fs, 'flattop', scaling='density')

f = f[1:]
PSD = PSD[:-1]

f_plt = np.linspace(np.amin(f), np.amax(f), 1000)
beta = 8 * np.log(2) * f_plt / np.pi**2

# parameters for plotting
plt.figure(1, figsize=(7, 4))
plt.title('Frequency Noise Power Spectral Density')
    
plt.xlabel(r'reference frequency $f$ (Hz)')
plt.ylabel(r'frequency noise PSD ($\rm{Hz}^2$ / Hz)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f, PSD)
plt.plot(f_plt, beta, ls='--', c='r')

plt.text(120, 8, r'$\beta$-separation', c='r', fontsize='large')  

#bbox=dict(facecolor='none', pad=0.3, boxstyle='round')

plt.xscale('log')
plt.yscale('log')

plt.savefig(f'Output/NFS PSD lock.png', dpi=300, bbox_inches='tight')
plt.show()

#%%

it = np.zeros(len(f)-2)

to_it = PSD / (f**2)

for i in range(0, len(it)):
    
   it[i] =  sg.simpson(y=to_it[i+2:], x=f[i+2:])

inter = si.interp1d(it, f[2:])

lw = inter(1/np.pi)

print(lw)

# parameters for plotting
plt.figure(1, figsize=(7, 4))
plt.title('Frequency Noise PSD Linewidth Integral')
    
plt.xlabel(r'reference frequency $f$ (Hz)')
plt.ylabel(r'$S_f(f) linewidth integral$ ($\rm{Hz}^2$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f[2:], it)
plt.axhline(1/np.pi, ls='--', c='r', label=r'$1 / \pi$')
plt.yscale('log')

plt.legend()

plt.savefig(f'Output/linewidth integral.png', dpi=300, bbox_inches='tight')
plt.show()
