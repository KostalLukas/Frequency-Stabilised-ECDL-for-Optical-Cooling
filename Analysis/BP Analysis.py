# -*- coding: utf-8 -*-
"""
Bare Laser Diode Power Analysis v1.0 

Lukas Kostal, 19.1.2024, ICL
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so


def lin(x, m, c):
    y = m * x + c
    return y


def ana(filename, Ifit, n):
    I, P = np.loadtxt(f'Data/{filename}.csv', delimiter=',', unpack=True, skiprows=1)
    I_err = np.ones(len(I)) * 0.1
    P_err = P * 0.03
    
    idx = np.argmin(np.abs(I - Ifit))
    n += idx
    
    popt, pcov = so.curve_fit(lin, I[idx:n], P[idx:n], absolute_sigma=True, sigma=P_err[idx:n])
    perr = np.sqrt(np.diag(pcov))

    return I, P, I_err, P_err, popt, perr

def thr(popt, perr):
    Ith = - popt[1] / popt[0]
    Ith_err = Ith * np.sqrt((perr[0] / popt[0])**2 + (perr[1] / popt[1])**2)
    
    return Ith, Ith_err


Ifit = 36
n = 14

I1, P1, I1_err, P1_err, popt1, perr1 = ana('PI_1b', Ifit, n)
I2, P2, I2_err, P2_err, popt2, perr2 = ana('PI_2b', Ifit, n)
I3, P3, I3_err, P3_err, popt3, perr3 = ana('PI_3c', Ifit, n)

I1th, I1th_err = thr(popt1, perr1)
I2th, I2th_err = thr(popt2, perr2)
I3th, I3th_err = thr(popt3, perr3)

Ith = np.array([I1th, I2th, I3th])
Ith_err = np.array([I1th_err, I2th_err, I3th_err])

for i in range(0, 3):
    print(f'threshold current f{i+1:.0f} = {Ith[i]:.4f} ± {Ith_err[i]:.4f} mA')

I_plt = np.linspace(np.amin([I1th, I2th, I3th]), 200)

# parameters for plotting
plt.figure(1)
plt.title('LD Power against Current')
    
plt.xlabel(r'current $I$ (mA)')
plt.ylabel(r'power $P$ (mW)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(I1, P1, c='b', label='bare diode')
plt.plot(I2, P2, c='g', label='cavity misaligned')
plt.plot(I3, P3, c='r', label='cavity aligned')

plt.legend()

# save the plot
plt.savefig('Output/diode power.pdf', dpi=300, bbox_inches='tight')
plt.show() 

# parameters for plotting
plt.figure(1)
plt.title('LD Power against Current')
    
plt.xlabel(r'current $I$ (mA)')
plt.ylabel(r'power $P$ (mW)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(I_plt, lin(I_plt, *popt1), ls='--', c='b')
plt.plot(I_plt, lin(I_plt, *popt2), ls='--', c='g')
plt.plot(I_plt, lin(I_plt, *popt3), ls='--', c='r')

plt.errorbar(I1, P1, P1_err, I1_err, fmt='.-', c='b', capsize=5, label='bare diode')
plt.errorbar(I2, P2, P2_err, I2_err, fmt='.-', c='g', capsize=5, label='cavity misaligned')
plt.errorbar(I3, P3, P3_err, I3_err, fmt='.-', c='r', capsize=5, label='cavity aligned')

plt.xlim(28, 42)
plt.ylim(-0.1, 3)

plt.legend()

# save the plot
plt.savefig('Output/diode threshold.pdf', dpi=300, bbox_inches='tight')
plt.show() 