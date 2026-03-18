#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as math
from qutip import *
import os
if not os.path.exists("data"):
    os.makedirs("data")
    
# defining operators for expectation values

# WARNING: code does not handle well large values of \alpha, \beta!

# defining operator for matrix element (1,1)
def op11(dim,sigma,alpha,beta):
    i,j = 0,0
    opa = 0*qeye(dim)
    for i in range(0,dim*10):
        opa += (-sigma)**i*(create(dim)-np.conj(alpha))**i*(destroy(dim)-alpha)**i/math.factorial(i)
        i += 1
    while True:
        opanew = opa + (-sigma)**i*(create(dim)-np.conj(alpha))**i*(destroy(dim)-alpha)**i/math.factorial(i)
        if np.linalg.norm(opanew-opa)<1e-3:
            break
        else:
            opa = opanew
            i += 1
    opb = 0*qeye(dim)
    for j in range(0,dim*10):
        opb += (-sigma)**j*(create(dim)-np.conj(beta))**j*(destroy(dim)-beta)**j/math.factorial(j)
        j += 1
    while True:
        opbnew = opb + (-sigma)**j*(create(dim)-np.conj(beta))**j*(destroy(dim)-beta)**j/math.factorial(j)
        if np.linalg.norm(opbnew-opb)<1e-3:
            break
        else:
            opb = opbnew
            j += 1
    return tensor(opa,opb)


# defining operator for matrix element (2,2)
def op22(dim,sigma,alpha,beta):
    i,j = 0,0
    opa = 0*qeye(dim)
    for i in range(0,dim*10):
        opa += (-sigma)**i*(create(dim)-np.conj(alpha))**i*(destroy(dim)-alpha)**i/math.factorial(i)
        i += 1
    while True:
        opanew = opa + (-sigma)**i*(create(dim)-np.conj(alpha))**i*(destroy(dim)-alpha)**i/math.factorial(i)
        if np.linalg.norm(opanew-opa)<1e-3:
            break
        else:
            opa = opanew
            i += 1
    opa = (create(dim)-np.conj(alpha))*opa*(destroy(dim)-alpha)
    opb = 0*qeye(dim)
    for j in range(0,dim*10):
        opb += (-sigma)**j*(create(dim)-np.conj(beta))**j*(destroy(dim)-beta)**j/math.factorial(j)
        j += 1
    while True:
        opbnew = opb + (-sigma)**j*(create(dim)-np.conj(beta))**j*(destroy(dim)-beta)**j/math.factorial(j)
        if np.linalg.norm(opbnew-opb)<1e-3:
            break
        else:
            opb = opbnew
            j += 1
    opb = (create(dim)-np.conj(beta))*opb*(destroy(dim)-beta)
    return tensor(opa,opb)


# defining operator for matrix element (1,2)
def op12(dim,sigma,alpha,beta):
    i,j = 0,0
    opa = 0*qeye(dim)
    for i in range(0,dim*10):
        opa += (-sigma)**i*(create(dim)-np.conj(alpha))**i*(destroy(dim)-alpha)**i/math.factorial(i)
        i += 1
    while True:
        opanew = opa + (-sigma)**i*(create(dim)-np.conj(alpha))**i*(destroy(dim)-alpha)**i/math.factorial(i)
        if np.linalg.norm(opanew-opa)<1e-3:
            break
        else:
            opa = opanew
            i += 1
    opa = opa*(destroy(dim)-alpha)
    opb = 0*qeye(dim)
    for j in range(0,dim*10):
        opb += (-sigma)**j*(create(dim)-np.conj(beta))**j*(destroy(dim)-beta)**j/math.factorial(j)
        j += 1
    while True:
        opbnew = opb + (-sigma)**j*(create(dim)-np.conj(beta))**j*(destroy(dim)-beta)**j/math.factorial(j)
        if np.linalg.norm(opbnew-opb)<1e-3:
            break
        else:
            opb = opbnew
            j += 1
    opb = (create(dim)-np.conj(beta))*opb
    return tensor(opa,opb)

# defining NOON state, including dephasing & losses, and corresponding matrix elements for minor M2

# defining NOON state
def NOON(N, p, tau):
    dim = N+2
    k=0
    state = 0
    while k<=N:
        state += (math.comb(N,k)*tau**(N-k)*(1-tau)**k*tensor(basis(dim,N-k),basis(dim,0))
                  *(tensor(basis(dim,N-k),basis(dim,0)).dag())
                 +math.comb(N,k)*tau**(N)*(1-tau)**(N-k)*tensor(basis(dim,0),basis(dim,k))
                  *(tensor(basis(dim,0),basis(dim,k)).dag()))
        k += 1
    state += tau**N*((1-p)*tensor(basis(dim,N),basis(dim,0))*(tensor(basis(dim,0),basis(dim,N)).dag())
            + (1-p)*tensor(basis(dim,0),basis(dim,N))*(tensor(basis(dim,N),basis(dim,0)).dag()))
    state = state.unit()
    return state


# matrix element (1,1), i.e., expectation value of op11 wrt NOON state
def NOON11(N,sigma,alpha,beta,p,tau):
    ans = expect(op11(N+2,sigma,alpha,beta), NOON(N,p,tau))
    return ans


# matrix element (2,2), i.e., expectation value of op22 wrt NOON state
def NOON22(N,sigma,alpha,beta,p,tau):
    ans = expect(op22(N+2,sigma,alpha,beta), NOON(N,p,tau))
    return ans


# matrix element (1,2), i.e., expectation value of op12 wrt NOON state
def NOON12(N,sigma,alpha,beta,p,tau):
    ans = expect(op12(N+2,sigma,alpha,beta), NOON(N,p,tau))
    return ans


# lowest-order minor for NOON state; one sigma with \sigma_i = \sigma_j
def NOONminor1(N,sigma,alpha,beta,p,tau):
    ans = (NOON11(N,sigma,alpha,beta,p,tau)*NOON22(N,sigma,alpha,beta,p,tau)-abs(NOON12(N,sigma,alpha,beta,p,tau))**2).real
    return ans

# generating data for negative regions of Husimi-based minor for various N

# DATA-GENERATING CODE; SLOW

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

NOONminor1vec = np.vectorize(NOONminor1)

Z3 = NOONminor1vec(3,1,X3,Y3,0,1)
Z3[Z3>=0]=np.nan #clipping points geq 0
np.savetxt("data/N3.csv", Z3, delimiter=",")

Z4 = NOONminor1vec(4,1,X4,Y4,0,1)
Z4[Z4>=0]=np.nan #clipping points geq 0
np.savetxt("data/N4.csv", Z4, delimiter=",")

Z5 = NOONminor1vec(5,1,X5,Y5,0,1)
Z5[Z5>=0]=np.nan #clipping points geq 0
np.savetxt("data/N5.csv", Z5, delimiter=",")

# data for negative values of minor as a function of \sigma at \alpha_opt, \beta_opt, optimized for the Husimi-based criterion
# finding coordinates of minimum for Husimi-based minor done in Mathematica

#range of \sigma
x = np.linspace(0.5,1,200) # [0.6,1] is sufficient as minor takes on +ve value for sigma<0.6

# defining optimal \alpha and \beta
minalpha2, minbeta2 = - 0.567293 - 1j*0.422113, + 0.422113 - 1j*0.567293
minalpha3, minbeta3 = + 0.158236 - 1j*0.987401, + 0.934232 - 1j*0.356665
minalpha4, minbeta4 = + 0.142056 + 1j*1.216480, + 0.960629 + 1j*0.759732
minalpha5, minbeta5 = + 0.254951 + 1j*1.391100, + 1.023850 + 1j*0.975489
minalpha6, minbeta6 = - 0.332335 + 1j*1.764830, - 0.824256 + 1j*0.959488
minalpha7, minbeta7 = + 0.877954 + 1j*1.001560, + 1.759120 - 1j*0.950369
minalpha8, minbeta8 = + 0.611792 - 1j*1.275190, - 1.620300 + 1j*1.452280

minz2 = NOONminor1vec(2,x,minalpha2,minbeta2,0,1)
minz3 = NOONminor1vec(3,x,minalpha3,minbeta3,0,1)
minz4 = NOONminor1vec(4,x,minalpha4,minbeta4,0,1)
minz5 = NOONminor1vec(5,x,minalpha5,minbeta5,0,1)
minz6 = NOONminor1vec(6,x,minalpha6,minbeta6,0,1)
minz7 = NOONminor1vec(7,x,minalpha7,minbeta7,0,1)
minz8 = NOONminor1vec(8,x,minalpha8,minbeta8,0,1)

# defining and saving full plot array
Nminz = np.array([x,minz2,minz3,minz4,minz5,minz6,minz7,minz8])
np.savetxt("data/Nminz.csv", Nminz, delimiter=",")

# data for effects of dephasing and losses
#range of \tau
x = np.linspace(0.5,1,200)

#adding losses
NOONnoisy2 = NOONminor1vec(2,1,minalpha2,minbeta2,0,x)
NOONnoisy3 = NOONminor1vec(3,1,minalpha3,minbeta3,0,x)
NOONnoisy4 = NOONminor1vec(4,1,minalpha4,minbeta4,0,x)
NOONnoisy5 = NOONminor1vec(5,1,minalpha5,minbeta5,0,x)
NOONnoisy6 = NOONminor1vec(6,1,minalpha6,minbeta6,0,x)
NOONnoisy7 = NOONminor1vec(7,1,minalpha7,minbeta7,0,x)
NOONnoisy8 = NOONminor1vec(8,1,minalpha8,minbeta8,0,x)

# defining and saving full plot array
NOONnoisy = np.array([x,NOONnoisy2,NOONnoisy3,NOONnoisy4,NOONnoisy5,NOONnoisy6,NOONnoisy7,NOONnoisy8])
np.savetxt("data/NOONnoisy.csv", NOONnoisy, delimiter=",")

print("done")