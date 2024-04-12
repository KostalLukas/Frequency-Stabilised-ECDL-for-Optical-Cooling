# -*- coding: utf-8 -*-
"""
Laser PI Curve Analysis v1.0 

Lukas Kostal, 3.4.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.interpolate as si


def lin(x, m, c):
    y = m * x + c
    return y


def ldat(filename):
    path = '/Users/lukaskostal/Desktop/Project/Data'
    I, P = np.loadtxt(f'{path}/{filename}.csv', delimiter=',', unpack=True, skiprows=1)
    I_err = np.ones(len(I)) * 0.1
    P_err = P * 0.03

    return I, P, I_err, P_err


I1, P1, I1_err, P1_err = ldat('PI_1b')
I2, P2, I2_err, P2_err = ldat('PI_2b')
I3, P3, I3_err, P3_err = ldat('PI_3c')

inter1 = si.interp1d(I1, P1, fill_value="extrapolate")
inter2 = si.interp1d(I2, P2, fill_value="extrapolate")
inter3 = si.interp1d(I3, P3, fill_value="extrapolate")

Ib_i = 42

Ib = np.linspace(Ib_i, np.amax(I1), 100)
b12 = inter2(Ib) / inter1(Ib)
b13 = inter3(Ib) / inter1(Ib)

b12_avg = np.mean(b12)
b12_sem = np.std(b12) / np.sqrt(len(b12))

print(f'output fraction = {b12_avg:.4f} ± {b12_sem:.4f}')
print(f'output fraction = {1-b12_avg:.4f} ± {b12_sem:.4f}')

# parameters for plotting
plt.figure(1, figsize=(6, 4))
plt.title('Operating Range PI Curve')
    
plt.xlabel(r'current $I$ (mA)')
plt.ylabel(r'output power $P$ (mW)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.plot(I1, P1, c='C0', label='bare laser diode')
plt.plot(I2, P2, c='C2', label='misaligned external cavity')
plt.plot(I3, P3, c='C3', label='aligned external cavity')

plt.legend()

plt.xlim(0, 180)

plt.savefig(f'Output/PI range.png', dpi=300, bbox_inches='tight')
plt.show()

# parameters for plotting
plt.figure(2, figsize=(6, 4))
plt.title('Lasing Threshold PI Curve')
    
plt.xlabel(r'current $I$ (mA)')
plt.ylabel(r'output power $P$ (mW)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.errorbar(I1, P1, xerr=I1_err, yerr=P1_err, capsize=5, c='C0', label='bare laser diode')
plt.errorbar(I2, P2, xerr=I2_err, yerr=P2_err, capsize=5, c='C2', label='misaligned external cavity')
plt.errorbar(I3, P3, xerr=I3_err, yerr=P3_err, capsize=5, c='C3', label='aligned external cavity')

plt.legend()

plt.xlim(28, 40)
plt.ylim(0, 4)

plt.savefig(f'Output/PI threshold.png', dpi=300, bbox_inches='tight')
plt.show()

