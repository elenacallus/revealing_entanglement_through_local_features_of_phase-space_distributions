#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as math
from qutip import *
import os
if not os.path.exists("data"):
    os.makedirs("data")
    

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


# defining random number state
def rand(d):
    dim = d+1
    i, j = 0, 0
    state = 0*tensor(basis(dim,0),basis(dim,0))
    while i<d:
        while j<d:
            state += (np.sqrt(np.random.uniform(0, 1)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi)))*tensor(basis(dim,i),basis(dim,j))
            j += 1
        i += 1
    state = state.unit()
    return state


# matrix element (1,1), i.e., expectation value of op11 wrt random number state
def rand11(d,sigma,alpha,beta):
    ans = expect(op11(d+1,sigma,alpha,beta), rand(d))
    return ans


# matrix element (2,2), i.e., expectation value of op11 wrt random number state
def rand22(d,sigma,alpha,beta):
    ans = expect(op22(d+1,sigma,alpha,beta), rand(d))
    return ans


# matrix element (1,2), i.e., expectation value of op11 wrt random number state
def rand12(d,sigma,alpha,beta):
    ans = expect(op12(d+1,sigma,alpha,beta), rand(d))
    return ans


# lowest-order minor for random number state; one sigma with sigma_i = sigma_j
def randminor1(d,sigma,alpha,beta):
    ans = (rand11(d,sigma,alpha,beta)*rand22(d,sigma,alpha,beta)-abs(rand12(d,sigma,alpha,beta))**2).real
    return ans


# lowest-order minor for random number state; two sigmas
def randminor2(d,sigma1,sigma2,alpha,beta):
    sigma11 = sigma1 + sigma1 - sigma1*sigma1
    sigma12 = sigma1 + sigma2 - sigma1*sigma2
    sigma22 = sigma2 + sigma2 - sigma2*sigma2
    ans = (rand11(d,sigma11,alpha,beta)*rand22(d,sigma22,alpha,beta)-abs(rand12(d,sigma12,alpha,beta))**2).real
    return ans

# generating data for random number state statistics
# WARNING: TAKES A LONG TIME

# values of sigma
x = np.linspace(0,1,20)
np.savetxt("data/random_x.csv", x, delimiter=",")

# defining number of shots for each width-parameter point
shots = 500

# range of dimensions from 2 to d-1
dim = 6
y11=np.zeros((dim-2,len(x)))
y22=np.zeros((dim-2,len(x)))

# at (\alpha,\beta) = (1,1)
for d in range(2,dim):
    i=0
    for i in range(len(x)):
        count, neg = 0, 0
        while count in range (0,shots):
            ans = randminor1(d,x[i],1,1)
            if ans<0:
                neg += 1
            count += 1
        y11[d-2,i] = neg/count
    np.savetxt("data/random_11.csv", y11, delimiter=",")

# at (\alpha,\beta) = (2,2)
for d in range(2,6):
    i=0
    for i in range(len(x)):
        count, neg = 0, 0
        while count in range (0,shots):
            ans = randminor1(d,x[i],2,2)
            if ans<0:
                neg += 1
            count += 1
        y22[d-2,i] = neg/count
    np.savetxt("data/random_22.csv", y22, delimiter=",")
    
    
print("done")