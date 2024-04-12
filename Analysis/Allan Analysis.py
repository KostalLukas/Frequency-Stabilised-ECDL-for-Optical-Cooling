# -*- coding: utf-8 -*-
"""
Allan Analysis v1.0

Lukas Kostal, 9.4.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.optimize as so
import allantools as at


def fit(x, x0, A, gam, c):
    y = A * gam * (x - x0) / ((x - x0)**2 + gam**2)**2 + c
    return y


def lin(x, m, c):
    y = c * x**m
    return y


fsr = 1.5e9

path_cal = "/Users/lukaskostal/Desktop/Project/Data/error calibration.CSV"
path_mes = "/Users/lukaskostal/Desktop/Project/Data/long error signal_1.CSV"

ds_cal = np.loadtxt(path_cal, unpack=True, delimiter=',', skiprows=2)
ds_mes = np.loadtxt(path_mes, unpack=True, delimiter=',', skiprows=2)

t = ds_cal[0, :]

t -= t[0]

for i in range(1, len(ds_cal[:, 0])):
    plt.plot(t, ds_cal[i, :])
plt.show()

ti = 9
tf = 20

idx_i = np.argmin(np.abs(t - ti))
idx_f = np.argmin(np.abs(t - tf))

t = t[idx_i:idx_f]
V_sig = ds_cal[3, idx_i:idx_f]
V_fpe = ds_cal[4, idx_i:idx_f]

I_fpe = (V_fpe - np.amin(V_fpe)) / np.ptp(V_fpe)

pks, _ = ss.find_peaks(I_fpe, height=0.9, prominence=0.5, width=10)

plt.plot(t, V_sig)
plt.plot(t, I_fpe)
plt.plot(t[pks], I_fpe[pks], 'x')
plt.show()

cal_tf = fsr / (t[pks[1]] - t[pks[0]])

ti = 9.5
tf = 11

idx_i = np.argmin(np.abs(t - ti))
idx_f = np.argmin(np.abs(t - tf))

t = t[idx_i:idx_f]
V_sig = V_sig[idx_i:idx_f]

ig = np.zeros(4)
ig[0] = t[int((np.argmin(V_sig) + np.argmax(V_sig)) / 2)]
ig[1] = np.amax(V_sig)
ig[2] = 0.1

opt, cov = so.curve_fit(fit, t, V_sig, p0=ig)

gam = opt[2]

cal_vf =  cal_tf * (2 / np.sqrt(3) * gam) / np.ptp(fit(t, *opt))

plt.plot(t, V_sig)
plt.plot(t, fit(t, *opt))
plt.show()

print(f'frequency calibration = {cal_vf * 1e-6:.6f} MHz V^-1')

#%%

t = ds_mes[0, :]
V_err = ds_mes[3, :]

t -= t[0]

plt.plot(t, V_err)
plt.show()

ti = 2
tf = 740

idx_i = np.argmin(np.abs(t - ti))
idx_f = np.argmin(np.abs(t - tf))

t = t[idx_i : idx_f]
f_err = V_err[idx_i : idx_f] * cal_vf

plt.plot(t, f_err)
plt.show()

fs = 1 / np.mean(np.diff(t))


out_ad = at.adev(f_err, fs, data_type='freq', taus='all')
t_ad = out_ad[0]
f_ad = out_ad[2]

ti = 0
tf = 1e-1
idx_i = np.argmin(np.abs(t_ad - ti))
idx_f = np.argmin(np.abs(t_ad - tf))
opt_1, cov_1 = so.curve_fit(lin, t_ad[idx_i:idx_f], f_ad[idx_i:idx_f], p0=[-1, 1])

ti = 2e0
tf = 9e0
idx_i = np.argmin(np.abs(t_ad - ti))
idx_f = np.argmin(np.abs(t_ad - tf))
opt_2, cov_2 = so.curve_fit(lin, t_ad[idx_i:idx_f], f_ad[idx_i:idx_f], p0=[1/2, -1])

ti = 2e1
tf = 1e3
idx_i = np.argmin(np.abs(t_ad - ti))
idx_f = np.argmin(np.abs(t_ad - tf))
opt_3, cov_3 = so.curve_fit(lin, t_ad[idx_i:idx_f], f_ad[idx_i:idx_f], p0=[2, -1])

t_plt = np.linspace(t_ad[0], t_ad[-1], 1000)

# parameters for plotting
plt.figure(1, figsize=(7, 5))
plt.title('Frequency Allan Deviation')
    
plt.xlabel(r'cluster time $\tau$ (s)')
plt.ylabel(r'Allan deviation $\sigma_{\rm{ADEV}}$ (Hz)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(t_ad, f_ad, c='b')
plt.plot(t_plt, lin(t_plt, *opt_1), c='C1', ls='--', label='white noise')
plt.plot(t_plt, lin(t_plt, *opt_2), c='C2', ls='--', label='random walk')
plt.plot(t_plt, lin(t_plt, *opt_3), c='C0', ls='--', label='drift')

plt.xscale('log')
plt.yscale('log')

plt.ylim(1, 6e2)

plt.legend(loc=0)

plt.savefig(f'Output/Allan deviation.png', dpi=300, bbox_inches='tight')
plt.show()