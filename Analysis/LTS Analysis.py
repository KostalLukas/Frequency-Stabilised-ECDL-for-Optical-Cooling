# -*- coding: utf-8 -*-
"""
Long Term Stability Analysis v1.0

Lukas Kostal, 19.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss
import scipy.optimize as so
import allantools as at


path_lock = '/Users/lukaskostal/Desktop/Project/Data/10min lock stability_1.lta'
path_free = "/Users/lukaskostal/Desktop/Project/Data/10min free stability_1.lta"

ch = 7

t_bck = 600

c_air = 299702547

df_lock = pd.read_table(f'{path_lock}', encoding='unicode_escape', skiprows=135)
df_free = pd.read_table(f'{path_free}', encoding='unicode_escape', skiprows=135)

t_lock = df_lock.iloc[:, 0].to_numpy()
t_lock *= 1e-3

idx = np.argmin(np.abs(t_lock - t_lock[-1] + t_bck))
t_lock = t_lock[idx:]
wl_lock = df_lock.iloc[idx:, ch].to_numpy()

idx = np.where(~np.isnan(wl_lock))
t_lock = t_lock[idx]
wl_lock = wl_lock[idx]

t_lock -= np.amin(t_lock)
wl_lock *= 1e-9
f_lock = c_air / wl_lock

t_free = df_free.iloc[:, 0].to_numpy()
t_free *= 1e-3

idx = np.argmin(np.abs(t_free - t_free[-1] + t_bck))
t_free = t_free[idx:]
wl_free = df_free.iloc[idx:, ch].to_numpy()

idx = np.where(~np.isnan(wl_free))
t_free = t_free[idx]
wl_free = wl_free[idx]

t_free -= np.amin(t_free)
wl_free *= 1e-9
f_free = c_air / wl_free

f_mean = np.mean(f_lock)
wn_mean = np.mean(1 / wl_lock)
wl_mean = np.mean(wl_lock)

print(f'mean lock frequency = {f_mean*1e-12} THz')
print(f'mean lock wavenumber = {wn_mean*1e2} cm^-1')

# parameters for plotting
plt.figure(1)
plt.title('Measured Frequency Stability')
    
plt.xlabel(r'time $t$ (s)')
plt.ylabel(r'frequency deviation $\delta f$ (MHz)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(t_lock, (f_lock-f_mean)*1e-6, c='C0')
plt.axhline(0, ls='--', c='r')

plt.legend()

plt.savefig(f'Output/LTS lock freq.png', dpi=300, bbox_inches='tight')
plt.show()


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(11, 5))

ax1.set_title('Frequency Drift Over 10 min')

ax1.plot(t_free, (f_free-np.mean(f_free))*1e-6, c='C1', label='free')
ax2.plot(t_lock, (f_lock-np.mean(f_lock))*1e-6, c='C0', label='locked')

ax1.legend(loc=4)
ax2.legend(loc=4)

fig.text(0.5, 0.06, 'time $t$ (s)', ha='center')
fig.text(0.06, 0.5, 'frequency deviation $\delta f$ (MHz)', va='center', rotation='vertical')
    
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
ax1.grid()
ax2.grid()
    
fig.subplots_adjust(hspace=0.06)

plt.savefig(f'Output/LTS stability.png', dpi=300, bbox_inches='tight')
plt.show()