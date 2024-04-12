# -*- coding: utf-8 -*-
"""
Mode-hop Free Scanning Range Analysis v1.0

Lukas Kostal, 17.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss
import scipy.optimize as so


def lin(x, m, c):
    y = m * x + c
    return y


path = '/Users/lukaskostal/Desktop/Project/Data/MHF WM_3.lta'

ch = 7

ti = 12955
tf = 12980

c_air = 299702547

df = pd.read_table(f'{path}', encoding='unicode_escape', skiprows=135)

t = df.iloc[:, 0].to_numpy()
t *= 1e-3

idxi = np.argmin(np.abs(t - ti))
idxf = np.argmin(np.abs(t - tf))

t = t[idxi:idxf]
wl = df.iloc[idxi:idxf, ch].to_numpy()

idx = np.where(~np.isnan(wl))

t = t[idx]
wl = wl[idx]

t -= np.amin(t)

f = c_air / (wl * 1e-9)
f -= np.amin(f)

pks_p, _ = ss.find_peaks(f)
pks_n, _ = ss.find_peaks(-f)

pks = np.concatenate((pks_p, pks_n))
pks = np.sort(pks)

t_res = t[pks[0]:pks[-1]]
f_res = np.zeros(len(t_res))

for i in range(0, len(pks)-1):
    f_fit = f[pks[i] : pks[i+1]]
    t_fit = t[pks[i] : pks[i+1]]
    
    popt, pcov = so.curve_fit(lin, t_fit, f_fit)
    
    f_res[pks[i]-pks[0] : pks[i+1]-pks[0]] = lin(t_fit, *popt)
    
f_res -= f[pks[0] : pks[-1]] 

t_rev = t_res[np.argwhere(np.abs(np.diff(f_res)) > 1e9)]
t_rev = t_rev[:, 0]
t_rev += np.mean(np.diff(t_res)) / 2


mhfr = np.amax(np.diff(f[pks]))
mhfr *= 1e-9

print(f'mode-hop free range mhfr = {mhfr:.3f} GHz')

# parameters for plotting
plt.figure(1, figsize=(6, 4))
plt.title('Mode-hop-free Range')

plt.xlabel(r'time $t$ (s)')
plt.ylabel(r'scan frequency $f$ (GHz)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(t, f*1e-9, c='C0')
plt.plot(t[pks_p], f[pks_p]*1e-9, 'x', c='r')
plt.plot(t[pks_n], f[pks_n]*1e-9, 'x', c='b')

plt.savefig(f'Output/MHF range WM.png', dpi=300, bbox_inches='tight')
plt.show()

# parameters for plotting
plt.figure(1)
plt.title('Nonlinearity in Frequency Scan')
    
plt.xlabel(r'time $t$ (s)')
plt.ylabel(r'frequency $f$ (MHz)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(t_res, f_res*1e-9, c='C0')

for i in range(0, len(t_rev)):
    plt.axvline(t_rev[i], ls='--', c='r')

plt.savefig(f'Output/MHF nonlinearity.png', dpi=300, bbox_inches='tight')
plt.show()

#%%

path = '/Users/lukaskostal/Desktop/Project/Data/MHF FPE_3.CSV'

fsr = 1.5e9

ti = 9
tf = 23

t1 = 1.7
t2 = 6.95

ds = np.loadtxt(f'{path}', delimiter=',', unpack=True, comments='#', skiprows=1)

t = ds[0, :]
V = ds[1, :]

t -= np.amin(t)

idxi = np.argmin(np.abs(t - ti))
idxf = np.argmin(np.abs(t - tf))

t = t[idxi:idxf] - t[idxi]
V = V[idxi:idxf]

I = V / np.amax(V)

pks, _ = ss.find_peaks(I, height=0.2, width=4)

# parameters for plotting
plt.figure(1, figsize=(6, 4))
plt.title('Mode-hop-free Range Etalon Transmission')
    
plt.xlabel(r'time $t$ (s)')
plt.ylabel(r'normalised intensity $I$ (arb.u.)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(t, I, c='C0')
plt.axvline(t1, ls='--', c='r')
plt.axvline(t2, ls='--', c='r')

plt.savefig(f'Output/MHF range FPE.png', dpi=300, bbox_inches='tight')
plt.show()
