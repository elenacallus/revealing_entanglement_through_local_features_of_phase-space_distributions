#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
import os
if not os.path.exists("plots"):
    os.makedirs("plots")


# plotting random number state statistics

# values of sigma
x = np.genfromtxt("data/random_x.csv", delimiter=',')
y11 = np.genfromtxt("data/random_11.csv", delimiter=',')
y22 = np.genfromtxt("data/random_22.csv", delimiter=',')
dim = len(y11)+2

cmap = plt.get_cmap('plasma')
colours = ['royalblue','olivedrab','darkorange','palevioletred']

fig, ax = plt.subplots()

# at (\alpha,\beta) = (1,1)
for d in range(2,dim): # 6
    ax.plot(x,100*y11[d-2], color=colours[d-2], ls="--")

# at (\alpha,\beta) = (2,2)
for d in range(2,dim): # 6
    ax.plot(x,100*y22[d-2],label=r'$d= %i$'%d, color=colours[d-2])

ax.set_title(r'$\textrm{- - - }\alpha=\beta=1 \quad \textrm{\textbf{----- }}\alpha=\beta=2$', pad=12, fontsize=22)
ax.set_xlabel(r'$\sigma$',fontsize=22)
ax.set_ylabel(r'$\% \textrm{ witnessed}$',fontsize=22,labelpad=5)
ax.tick_params(labelsize=20)
ax.set_axisbelow(True)
ax.xaxis.grid(ls='dashed',alpha=0.75)
ax.yaxis.grid(which='major',ls='dashed',alpha=0.75)
ax.yaxis.grid(which='minor',ls='dashed',alpha=0.75)
ax.set_xlim([0,1])
ax.set_yticks([0,20,40,60])
ax.set_ylim([0, 60])
plt.legend(loc=2,fontsize=14)
plt.savefig('plots/randomnumber.pdf',dpi=600,bbox_inches='tight')
plt.show()

print("done")