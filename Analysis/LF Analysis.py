# -*- coding: utf-8 -*-
"""
Locking Feature Analysis v1.0

Lukas Kostal, 18.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so


def lin(x, m, c):
    y = m * x + c
    return y


path = '/Users/lukaskostal/Desktop/Project/Data/lock transition_1.CSV'

cal_vf = 2031089517.2731104

data = np.loadtxt(f'{path}', delimiter=',', unpack=True, comments='#', skiprows=1)

t = data[0, :]
t -= t[0]

V_dfs = data[3, :]
V_rmp = data[2, :]
V_rmp -= np.amin(V_rmp)

plt.plot(t, V_dfs)
plt.plot(t, V_rmp)
plt.show()

ti = 4.7
tf = 7

idx_i = np.argmin(np.abs(t - ti))
idx_f = np.argmin(np.abs(t - tf))

t = t[idx_i : idx_f]
V_dfs = V_dfs[idx_i : idx_f]
V_rmp = V_rmp[idx_i : idx_f]

Vs_rmp = V_rmp[np.argmax(V_dfs) : np.argmin(V_dfs)]
Vs_dfs = V_dfs[np.argmax(V_dfs) : np.argmin(V_dfs)]

V_ofst = Vs_rmp[np.argmin(np.abs(Vs_dfs))]

popt, pcov = so.curve_fit(lin, t, V_rmp)

f = cal_vf * lin(t, *popt)

# parameters for plotting
plt.figure(1, figsize=(7, 4))
plt.title('Frequency Lock Spectroscopy Feature')
    
plt.xlabel(r'applied voltage (V)')
plt.ylabel(r'lock-in signal (V)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(f*1e-9, V_dfs)

plt.savefig(f'Output/DFS lock transition.png', dpi=300, bbox_inches='tight')
plt.show()