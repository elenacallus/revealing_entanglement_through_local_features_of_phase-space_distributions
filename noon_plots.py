#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from labellines import labelLines
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
import os
if not os.path.exists("plots"):
    os.makedirs("plots")
    

# plotting negative regions of Husimi-based minor for various N

# N=3
x3 = np.linspace(0,1.5,100)
y3 = np.linspace(-1.5,0,100)
X3,Y3 = np.meshgrid(x3,y3)

# N=4
x4 = np.linspace(1.5,3,100)
y4 = np.linspace(1.5,3,100)
X4,Y4 = np.meshgrid(x4,y4)

# N=5
x5 = np.linspace(0,2.5,100)
y5 = np.linspace(-2.5,0,100)
X5,Y5 = np.meshgrid(x5,y5)

Z3 = np.genfromtxt("data/N3.csv", delimiter=',')

Z4 = np.genfromtxt("data/N4.csv", delimiter=',')

Z5 = np.genfromtxt("data/N5.csv", delimiter=',')

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.45))

#first plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 1)
c1 = ax.pcolormesh(X3, Y3, Z3, cmap=cm.magma, vmax=0)
ax.set_title(r'$N=3$', fontsize=12, pad=10)
ax.set_ylabel(r'$\beta$',fontsize=12,labelpad=10)
ax.set_xlabel(r'$\alpha$',fontsize=12)
ax.tick_params(labelsize=12)
ax.set_xticks([0,0.5,1.0,1.5])
ax.set_yticks([-1.5,-1.0,-0.5,0])
cbar = plt.colorbar(c1,ax=ax,location='bottom',pad=0.15)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.tick_params(labelsize=12)
cbar.ax.xaxis.get_offset_text().set_fontsize(12)

#second plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 2)
c2=ax.pcolormesh(X4, Y4, Z4, cmap=cm.magma, vmax=0)
ax.set_title(r'$N=4$', fontsize=12, pad=10)
ax.set_xlabel(r'$\alpha$',fontsize=12)
ax.tick_params(labelsize=12)
ax.set_xticks([1.5,2.0,2.5,3.0])
ax.set_yticks([1.5,2.0,2.5,3.0])
cbar = plt.colorbar(c2,ax=ax,location='bottom',pad=0.15)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.tick_params(labelsize=12)
cbar.ax.xaxis.get_offset_text().set_fontsize(12)

#third plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 3)
c3=ax.pcolormesh(X5, Y5, Z5, cmap=cm.magma, vmax=0)
ax.set_title(r'$N=5$', fontsize=12, pad=10)
ax.set_xlabel(r'$\alpha$',fontsize=12)
ax.tick_params(labelsize=12)
ax.set_xlim([0.5,2])
ax.set_ylim([-2, -0.5])
ax.set_xticks([0.5,1.0,1.5,2.0])
ax.set_yticks([-2,-1.5,-1,-0.5])
cbar = plt.colorbar(c3,ax=ax,location='bottom',pad=0.15)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.tick_params(labelsize=12)
cbar.ax.xaxis.get_offset_text().set_fontsize(12)

plt.savefig('plots/NOON_Husimi_345.png',dpi=600,bbox_inches='tight')
plt.show()


# plotting negative values of minor as a function of \sigma at \alpha_opt, \beta_opt, optimized for the Husimi-based criterion

x = np.genfromtxt("data/Nminz.csv", delimiter=',')[0]
minz2 = np.genfromtxt("data/Nminz.csv", delimiter=',')[1]
minz3 = np.genfromtxt("data/Nminz.csv", delimiter=',')[2]
minz4 = np.genfromtxt("data/Nminz.csv", delimiter=',')[3]
minz5 = np.genfromtxt("data/Nminz.csv", delimiter=',')[4]
minz6 = np.genfromtxt("data/Nminz.csv", delimiter=',')[5]
minz7 = np.genfromtxt("data/Nminz.csv", delimiter=',')[6]
minz8 = np.genfromtxt("data/Nminz.csv", delimiter=',')[7]


fig, ax = plt.subplots()

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(1, 0, 8)]

plt.plot(x, minz2, color=colors[1], label=r'$\boldsymbol{ N= 2}$')
plt.plot(x, minz3, color=colors[2], label=r'$\boldsymbol{ N= 3}$')
plt.plot(x, minz4, color=colors[3], label=r'$\boldsymbol{ N= 4}$')
plt.plot(x, minz5, color=colors[4], label=r'$\boldsymbol{ N= 5}$')
plt.plot(x, minz6, color=colors[5], label=r'$\boldsymbol{ N= 6}$')
plt.plot(x, minz7, color=colors[6], label=r'$\boldsymbol{ N= 7}$')
plt.plot(x, minz8, color=colors[7], label=r'$\boldsymbol{ N= 8}$')
lines = plt.gca().get_lines()
labelLines(lines, align=False,xvals=np.linspace(0.95,0.95,8),fontsize=18,color='black')
ax.set_xlabel(r'$\sigma$',fontsize=22)
ax.set_ylabel(r'$M_2(\alpha_\textrm{opt},\beta_\textrm{opt};\sigma)$',fontsize=22,labelpad=5)
ax.tick_params(labelsize=20)
ax.set_axisbelow(True)
ax.xaxis.grid(ls='dashed',alpha=0.75)
ax.yaxis.grid(which='major',ls='dashed',alpha=0.75)
ax.yaxis.grid(which='minor',ls='dashed',alpha=0.75)
ax.set_yscale('symlog',linthresh=1e-6)
ax.set_xlim([0.5,1.0])
ax.set_ylim([None, -1e-6])
plt.savefig('plots/NOON_varsigma.pdf',dpi=300,bbox_inches='tight')
plt.show()


# plotting effects of dephasing and losses

x = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[0]
NOONnoisy2 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[1]
NOONnoisy3 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[2]
NOONnoisy4 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[3]
NOONnoisy5 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[4]
NOONnoisy6 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[5]
NOONnoisy7 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[6]
NOONnoisy8 = np.genfromtxt("data/NOONnoisy.csv", delimiter=',')[7]

# plotting

fig, ax = plt.subplots()

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(1, 0, 8)]

plt.plot(x, NOONnoisy2, color=colors[1], label=r'$\boldsymbol{ N= 2}$')
plt.plot(x, NOONnoisy3, color=colors[2], label=r'$\boldsymbol{ N= 3}$')
plt.plot(x, NOONnoisy4, color=colors[3], label=r'$\boldsymbol{ N= 4}$')
plt.plot(x, NOONnoisy5, color=colors[4], label=r'$\boldsymbol{ N= 5}$')
plt.plot(x, NOONnoisy6, color=colors[5], label=r'$\boldsymbol{ N= 6}$')
plt.plot(x, NOONnoisy7, color=colors[6], label=r'$\boldsymbol{ N= 7}$')
plt.plot(x, NOONnoisy8, color=colors[7], label=r'$\boldsymbol{ N= 8}$')
lines = plt.gca().get_lines()
labelLines(lines, align=False,xvals=np.linspace(1,1,8),fontsize=18,color='black')
ax.set_xlabel(r'$\tau$',fontsize=22)
ax.set_ylabel(r'$M_2(\alpha_\textrm{opt},\beta_\textrm{opt};1)$',fontsize=22,labelpad=5)
ax.tick_params(labelsize=20)
ax.set_axisbelow(True)
ax.xaxis.grid(ls='dashed',alpha=0.75)
ax.yaxis.grid(which='major',ls='dashed',alpha=0.75)
ax.yaxis.grid(which='minor',ls='dashed',alpha=0.75)
ax.set_yscale('symlog',linthresh=1e-6)
ax.set_xlim([0.5,1])
ax.set_ylim([None, -1e-6])
plt.savefig('plots/NOON_noisy.pdf',dpi=300,bbox_inches='tight')
plt.show()

print("done")