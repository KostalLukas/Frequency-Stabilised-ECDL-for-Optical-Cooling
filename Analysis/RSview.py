# -*- coding: utf-8 -*-
"""
RTB 2000 series oscilloscope data preview

Lukas Kostal, 15.3.2024, ICL
"""


import numpy as np
import matplotlib.pyplot as plt


path = input('input path of dataset to display \t')

if path[-1] == ' ':
    path = path[:-1]
    
path = path.replace('\ ', ' ')

data = np.loadtxt(f'{path}', delimiter=',', unpack=True, comments='#', skiprows=1)

t = data[0, :]
V = data[1:, :]

n = len(V[:, 0])

clr = ['darkorange', 'g', 'r', 'b']
nam = ['Ch1', 'Ch2', 'Ch3', 'Ch4']

# parameters for plotting
plt.figure(1)
plt.title(f'{path}')
    
plt.xlabel(r'time $t$ (s)')
plt.ylabel(r'signal (V)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

for i in range(0, n):
    plt.plot(t, V[i, :], c=clr[i], label=nam[i])

plt.legend()
    
plt.show()