# -*- coding: utf-8 -*-

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clrs
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
import os
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("plots"):
    os.makedirs("plots")
    

# defining normalisation constant
def norm(gamma,p):
    N = (2-2*(1-p)*np.exp(-4*abs(gamma)**2))**(-1)
    return N


# matrix element (1,1), i.e., expectation value of op11 wrt dephased cat state
def cat11(gamma,p,sigma,alpha):
    N = norm(gamma,p)
    ans = N*(np.exp(-2*sigma*abs(gamma-alpha)**2)+np.exp(-2*sigma*abs(-gamma-alpha)**2)
            -(1-p)*np.exp(-4*abs(gamma)**2)*(
                np.exp(-2*sigma*(-abs(gamma)**2+abs(alpha)**2+2*(np.conj(gamma)*alpha).imag))
                +np.exp(-2*sigma*(-abs(gamma)**2+abs(alpha)**2-2*(np.conj(gamma)*alpha).imag))
            ))
    return ans


# matrix element (2,2), i.e., expectation value of op22 wrt dephased cat state
def cat22(gamma,p,sigma,alpha):
    N = norm(gamma,p)
    ans = N*(abs(gamma-alpha)**4*np.exp(-2*sigma*abs(gamma-alpha)**2)
             +abs(-gamma-alpha)**4*np.exp(-2*sigma*abs(-gamma-alpha)**2)
            -(1-p)*np.exp(-4*abs(gamma)**2)*(
                (gamma-alpha)**2*np.conj(-gamma-alpha)**2*np.exp(-2*sigma*(-abs(gamma)**2+abs(alpha)**2+2*(np.conj(gamma)*alpha).imag))
                +(-gamma-alpha)**2*np.conj(gamma-alpha)**2*np.exp(-2*sigma*(-abs(gamma)**2+abs(alpha)**2-2*(np.conj(gamma)*alpha).imag))
            ))
    return ans


# matrix element (1,2), i.e., expectation value of op12 wrt dephased cat state
def cat12(gamma,p,sigma,alpha):
    N = norm(gamma,p)
    ans = N*(abs(gamma-alpha)**2*np.exp(-2*sigma*abs(gamma-alpha)**2)
             +abs(-gamma-alpha)**2*np.exp(-2*sigma*abs(-gamma-alpha)**2)
            -(1-p)*np.exp(-4*abs(gamma)**2)*(
                (gamma-alpha)*np.conj(-gamma-alpha)*np.exp(-2*sigma*(-abs(gamma)**2+abs(alpha)**2+2*(np.conj(gamma)*alpha).imag))
                +(-gamma-alpha)*np.conj(gamma-alpha)*np.exp(-2*sigma*(-abs(gamma)**2+abs(alpha)**2-2*(np.conj(gamma)*alpha).imag))
            ))
    return ans


# lowest-order minor for random number state; one sigma with \sigma_i = \sigma_j
# theta is the phase of the coordinate point alpha
def catminor1(gamma,p,sigma,alpha,theta):
    alpha = alpha*np.exp(1j*theta)
    ans = (cat11(gamma,p,sigma,alpha)*cat22(gamma,p,sigma,alpha)-abs(cat12(gamma,p,sigma,alpha))**2).real
    return ans


# plotting Husimi-based minor for alpha = gamma (linear scaling) and alpha = i*gamma (log scaling)

# range for theta=0
xcat1 = np.linspace(0.0001,2,200)
ycat1 = np.linspace(0,1,200)
Xcat1, Ycat1 = np.meshgrid(xcat1,ycat1)

# range for theta=pi/2
xcat2 = np.linspace(0.0001,5,200)
ycat2 = np.linspace(0.0001,5,200)
Xcat2, Ycat2 = np.meshgrid(xcat2,ycat2)

catminor1vec = np.vectorize(catminor1)

Zcat0 = catminor1vec(Xcat1,0,1,Ycat1,np.pi*0)
Zcat0[Zcat0>=0]=np.nan #clipping points geq 0, before: -1e-15
np.savetxt("data/cat_theta0.csv", Zcat0, delimiter=",")

Zcat90 = catminor1vec(Xcat2,0,1,Ycat2,np.pi*0.5)
Zcat90[Zcat0>=0]=np.nan #clipping points geq 0, before: -1e-15
np.savetxt("data/cat_theta90.csv", Zcat0, delimiter=",")

# set up a figure
fig = plt.figure(figsize=(4, 5.5), layout="constrained")

#first plot
# set up the Axes for the first plot
ax = fig.add_subplot(2, 1, 1)
c1 = ax.pcolormesh(Xcat1, Ycat1, Zcat0, cmap=cm.magma, vmax=0)
ax.set_ylabel(r'$|\alpha|$',fontsize=14,labelpad=5)
ax.tick_params(labelsize=12)
cbar = plt.colorbar(c1,ax=ax,location='right',pad=0.0)
cbar.ax.tick_params(labelsize=12)

ax.text(1.1, 0.08333, r'$\arg(\alpha) = \arg(\gamma)$', fontsize=11)

#second plot
# set up the Axes for the first plot
ax = fig.add_subplot(2, 1, 2)
c2=ax.pcolormesh(Xcat2, Ycat2, Zcat90, cmap=cm.magma,norm=clrs.SymLogNorm(linthresh=1e-10, vmin=np.nanmin(Zcat90), vmax=0))
cbar = plt.colorbar(c2,ax=ax,location='right',pad=0.03, ticks=[-1e1,-1e-2,-1e-5,-1e-8,0])
ax.set_ylabel(r'$|\alpha|$',fontsize=14,labelpad=5)
ax.set_xlabel(r'$|\gamma|$',fontsize=14)
ax.tick_params(labelsize=12)
cbar.ax.tick_params(labelsize=12)

ax.text(2.2, 0.41665, r'$\arg(\alpha) = \arg(\gamma)+\frac{\pi}{2}$', fontsize=11)

plt.savefig('plots/cat_theta090_linearlog.png',dpi=600,bbox_inches='tight')
plt.show()


# data for Husimi-based criterion with dephased state

xcat = np.linspace(0.00001,4,300)
ycat = np.linspace(0,1,300)
Xcat, Ycat = np.meshgrid(xcat,ycat)

Zcat_p = catminor1(Xcat,Ycat,1,Xcat,np.pi*0.5)
Zcat_p[Zcat_p>=0]=np.nan #clipping points geq 0

np.savetxt("data/cat_dephasing.csv", Zcat_p, delimiter=",")


# plotting Husimi-based minor as a function of p and gamma

fig, ax = plt.subplots()
c1 = ax.pcolormesh(Xcat, Ycat, Zcat_p, cmap=cm.magma, norm=clrs.SymLogNorm(linthresh=1e-3, vmin=np.nanmin(Zcat_p), vmax=0))
ax.set_xlabel(r'$|\gamma|$',fontsize=22)
ax.set_ylabel(r'$p$',fontsize=22,rotation=0,labelpad=14)
ax.tick_params(labelsize=20)
ax.set_xticks([0,1.0,2.0,3.0,4.0])
cbar = plt.colorbar(c1, ticks=[-1e2,-1e0,-1e-2,0])
cbar.ax.tick_params(labelsize=20)
plt.savefig('plots/cat_dephasing.png',dpi=600,bbox_inches='tight')
plt.show()


# data for minor as a function of gamma and sigma

xcat = np.linspace(0.0001,4,400)
sigmacat = np.linspace(0,1,400)
Xcat, Sigmacat = np.meshgrid(xcat,sigmacat)

Zcat_sigma = catminor1(Xcat,0,Sigmacat,Xcat,np.pi*0.5)
Zcat_sigma[Zcat_sigma>=-1e-10]=np.nan #clipping points geq 0

np.savetxt("data/cat1_sigma.csv", Zcat_sigma, delimiter=",")


# plotting minor as a function of gamma and sigma

fig, ax = plt.subplots()
c1 = ax.pcolormesh(Xcat, Sigmacat, Zcat_sigma, cmap=cm.magma, norm=clrs.SymLogNorm(linthresh=1e-10, vmin=np.nanmin(Zcat_sigma), vmax=0))
ax.set_title(r'$\alpha=\mathrm{i}\gamma$', pad=12, fontsize=22)
ax.set_xlabel(r'$|\gamma|$',fontsize=22)
ax.set_ylabel(r'$\sigma$',fontsize=22,rotation=0,labelpad=14)
ax.tick_params(labelsize=20)
cbar = plt.colorbar(c1, ticks=[-1e1,-1e-2,-1e-5,-1e-8,0])
cbar.ax.tick_params(labelsize=20)
plt.savefig('plots/cat_sigma.png',dpi=600,bbox_inches='tight')
plt.show()


print("done")