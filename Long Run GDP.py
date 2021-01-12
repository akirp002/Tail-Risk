#!/usr/bin/env python
# coding: utf-8

# In[75]:


import cupy as cp
import numpy as np
from cupy import random
import scipy as sc
from scipy import linalg
import matplotlib as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from sympy import Matrix
import seaborn as sns
import math
import pandas as pd
import scipy as sc
import time
y_raw = pd.read_csv(r'C:\Research\Tail Risk\Macro Data\Book2.csv', delimiter=',')
y_data = cp.zeros([T,5])
T = 133
J = 1000
m= cp.matmul
resh = cp.reshape
norm= cp.random.standard_normal


# In[ ]:


def initialize_SIG():
    SIG = cp.zeros([5,5,T])
    for j in range(5):
            SIG[j,j,:] = 1
    return SIG


# In[ ]:


def propogate_cov(SIG,ERR,l):
    J = 10000
    LOL = cp.zeros([T])
    d  = .01*cp.exp(cp.random.standard_normal([5,J]))
    h = .01*cp.random.standard_normal([10,J])
    H = cp.zeros([10,T-l])
    D = cp.zeros([5,T-l])
    x = cp.var(ERR[0,:],0)
    for t in range(T-l):
        eps = cp.random.standard_normal([5,J])
        zz = eps*d
        zz[0] =  zz[0]
        zz[1] =  zz[1]+ h[0]*eps[1]
        zz[2] =  zz[2]+ cp.sum(h[1:3]*eps[0:2,:],0)
        zz[3] =  zz[3]+ cp.sum(h[3:6]*eps[0:3,:],0)
        zz[4] =  zz[4]+ cp.sum(h[6:]*eps[0:4,:],0)

        Error= cp.sum((cp.reshape(ERR[:,t],[5,1]) - zz)**2,0);
        w =((1/(2*math.pi)**(-5)))*cp.exp(-(1/2)*Error)
        #w = Error;
        w[cp.where(cp.isnan(w))] =0 
        w = w/(cp.sum(w))
        idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
        h = h[:,idx]
        d = d[:,idx]
        d = cp.exp(d*.1*cp.random.standard_normal([5,J]))
        h = h*cp.random.standard_normal([10,J])
        H[:,t] = cp.mean(h,1)
        D[:,t] = cp.mean(d,1)

        SIG[:,:,t] = cp.vstack([cp.zeros([5]),
               cp.hstack([H[0:1,t],cp.zeros(4)]),
               cp.hstack([H[1:3,t],cp.zeros(3)]),
               cp.hstack([H[3:6,t],cp.zeros(2)]),
               cp.hstack([H[6:10,t],cp.zeros(1)])]) + cp.diag(D[:,t]**2) 
        
        SIG[:,:,t] = cp.tril(SIG[:,:,t]).T+cp.tril(SIG[:,:,t])-cp.diag(cp.diag(SIG[:,:,t])) 
        
    return  SIG


# In[ ]:


# B block
def propogate_B(A,B,tau,g,y_data,Sig_B,T,l,SIG):
    t = l
    P0  = 1*norm([5,5])
    y_hat = cp.zeros([5,1])
    while t <T:
        B[:,t] = B[:,t-1]
        q1 = cp.kron(cp.ones([5,1]),resh(y_data[t-1,:],[1,5*l]))
        q2 = cp.kron(cp.eye(5),cp.ones([1,l]))
        R = q1*q2
        P = Sig_B+P0
        D = m(m(R,P),R.T) +SIG[:,:,t-l]
        L = m(P,R.T)
        y_hat = resh((resh(A[:,t],[5,1]) +resh(m(R,B[:,t]),[5,1]) +resh(tau[:,t],[5,1])+cp.kron(g[:,t],cp.ones([5,1]))),[5])
        B[:,t] = B[:,t] + cp.matmul((m(L,cp.linalg.inv(D))),(d*cp.reshape(y_data[t,:],[5,1])-c*resh(y_hat,[5,1])))[:,0]     
        P0 = P - m(L,m(cp.linalg.inv(D),L.T))
        t = t+1
    return B


# In[ ]:


# A block
def propogate_A(A,B,tau,g,y_data,Sig_A,T,l,SIG):
    t = l
    P0  = 1*norm([5,5])
    y_hat = cp.zeros([5,1])
    while t <T:
        A[:,t] = A[:,t-1]
        P = Sig_A+P0
        D = P + SIG[:,:,t-l]
        L = P
        q1 = cp.kron(cp.ones([5,1]),resh(y_data[t-1,:],[1,5*l]))
        q2 = cp.kron(cp.eye(5),cp.ones([1,l]))
        R = q1*q2
        y_hat = resh((resh(A[:,t],[5,1]) +1*resh(m(R,B[:,t]),[5,1]) +resh(tau[:,t],[5,1])+cp.kron(g[:,t],cp.ones([5,1]))),[5])
        A[:,t] = A[:,t]+m((m(L,cp.linalg.inv(D))),(d*resh(y_data[t,:],[5,1])-c*resh(y_hat,[5,1])))[:,0]      
        P0 = P - m(L,m(cp.linalg.inv(D),L.T))
        t = t+1

    return A


# In[ ]:


def draw_covariance(ZZ,ndim):
    iw = sc.stats._multivariate.wishart_gen()
    V = cp.zeros([ndim,ndim])
    t=  l
    while t<T:
        V = cp.matmul(cp.reshape((ZZ[:,t]-ZZ[:,t-1]),[ndim,1]),cp.reshape((ZZ[:,t]-ZZ[:,t-1]),[1,ndim]))+V
        t = t+1
    
    V = iw.rvs(ndim,cp.asnumpy(cp.linalg.inv(10*cp.eye(ndim)+V)))
    
    return V


# In[ ]:


def power_law(k_min,y,alpha):
    gamma =-1-alpha
    return ((-k_min**(gamma+1))*y + k_min**(gamma+1.0))**(1.0/(gamma + 1.0))


# In[ ]:


def Gen_likic(y_data,A,B,tau,g,SIG,l):
    T = 133
    Y = cp.zeros([5,T])
    LIK = cp.zeros([T])
    error = cp.zeros([5,1,T])
    t = l
    while t<T:
        q1 = cp.kron(cp.ones([5,1]),resh(y_data[t-1,:],[1,5*l]))
        q2 = cp.kron(cp.eye(5),cp.ones([1,l]))
        R = q1*q2
        #Y[:,t] = resh((resh(A[:,t],[5,1]) +1*resh(m(R,B[:,t]),[5,1]) +resh(tau[:,t],[5,1])+cp.kron(g[:,t],cp.ones([5,1]))),[5])
        Y[:,t] = resh((resh(A[:,t],[5,1]) +1*resh(m(R,B[:,t]),[5,1]) +resh(tau[:,t],[5,1])+cp.kron(g[:,t],cp.ones([5,1]))),[5])

        error[:,:,t] = resh((d*y_data[t,:].T-c*Y[:,t]),[5,1])
        X = cp.linalg.inv(SIG[:,:,t])
        #LIK[t] = cp.float(cp.log((math.pi**-2.5)*abs(cp.linalg.det(SIG[:,:,t-l])**.5))+(-1/2)*cp.matmul(cp.matmul(error[:,:,t].T,X),error[:,:,t]))
        t = t+1 
    SSE = cp.sum(error**2)
    ERR = cp.reshape(error[:,:,l:T],[5,T-l])
    likic = cp.sum(LIK[l:T])
    return Y,likic,ERR,SSE


# In[ ]:


def initialize(l):
    T = 133
    A = .001*cp.random.standard_normal([5,T])
    B = .001*cp.random.standard_normal([5*l,T])
    Sig_A = 1*cp.diag(cp.random.standard_normal([5])**2)
    Sig_B =1*(cp.diag(cp.random.standard_normal([5,5])**2))

    tau =cp.array( (sc.stats.truncnorm.rvs(1.105,sc.inf,size = [5,J])**-5))
    g =cp.array((sc.stats.truncnorm.rvs(1.105,sc.inf,size = [J])**-5))
    
    #tau = 1*power_law(1.105,cp.random.uniform(0,1,[5,T]),5)
    #g =1* power_law(1.105,cp.random.uniform(0,1,[1,T]),5)
    g = cp.random.standard_normal([5,J])
    U = cp.random.uniform(0,1,[5,T])
    U_g = cp.random.uniform(0,1,[T]) 
    lamb = .1*cp.ones([5,1])
    lamb_g = .1
    tau[0,cp.where((lamb[0] - U[0])< 0)[0]] = 0 
    tau[1,cp.where((lamb[1] - U[1])< 0)[0]] = 0 
    tau[2,cp.where((lamb[2] - U[2])< 0)[0]] =0
    tau[3,cp.where((lamb[3] - U[3])< 0)[0]] =0
    tau[4,cp.where((lamb[4] - U[4])< 0)[0]] =0
    g[:,cp.where((lamb_g - U_g)< 0)[0]] = 0
    
    
    
    
    return A,Sig_A,tau,g ,Sig_B,B


# In[ ]:


l = 1
c = 1
d = 1
A,Sig_A,tau,g ,Sig_B,B= initialize(l)
SIG = initialize_SIG()
tau = tau
g = g
B = B
A = propogate_A(A,B,tau,g,y_data,Sig_A,T,l,SIG)
B = propogate_B(A,B,tau,g,y_data,Sig_B,T,l,SIG)
Y,likic,ERR,SSE = Gen_likic(y_data,A,B,tau,g,SIG,l)
print("error before:", SSE)
#plt.plot(cp.asnumpy(y_data[:,0]))
plt.plot(cp.asnumpy(Y[0,:]))


# In[ ]:


plt.plot(cp.asnumpy(y_data[:,0]))


# In[ ]:


plt.plot(cp.asnumpy(y_data[:,2]))
plt.plot(cp.asnumpy(Y[2,:]/d))


# In[ ]:


#def propogate_tau():
t = l
Z = cp.zeros([5,T])
tau_1 = cp.zeros([5,T]) 
tau_11 = cp.zeros([5,T]) 
lamb = .1
xx = 0
J = 10000

while t <T:        
    q = cp.random.uniform(high=1,size=[5,J])
    q = lamb -q
        
    #tau_j = power_law(1.105,cp.random.uniform(0,1,[5,J]),5)
    tau_j = (sc.stats.truncnorm.rvs(1.105,sc.inf,size = [5,J])**-5)
    Y[:,t]=   Y[:,t]-tau[:,t]
    LL0 =  cp.exp(-.5*(resh(y_data[t,:]-(1/d)*Y[:,t],[5,1]))**2 ) *resh(1-lamb,[5,1])
    Z = (1/d)*(resh(Y[:,t],[5,1]) + tau_j)
    LLJ = cp.exp(-.5*(resh(y_data[t,:],[5,1])-Z)**2) *resh(lamb,[5,1])
    for i in range(5):
        if LL0[i]>cp.mean(LLJ[i]):
                tau_1[i] = 0
        else:
            w = LLJ[i]/cp.sum(LLJ[i])
            tau_11[i,t]  = cp.sum(w*tau_j[i,:])
            print(tau_11[i,t])
            xx = xx+1

    
    

    t = t+1
print(xx)


# In[ ]:


l = 1
c = 1
d = 1
A,Sig_A,tau,g ,Sig_B,B= initialize(l)
SIG = initialize_SIG()
tau = tau
g = g*0
A = A*0
A = propogate_A(A,B,tau_11,g,y_data,Sig_A,T,l,SIG)
B = propogate_B(A,B,tau_11,g,y_data,Sig_B,T,l,SIG)
Y,likic,ERR,SSE = Gen_likic(y_data,A,B,tau_11,g,SIG,l)
print("error before:", SSE)
#plt.plot(cp.asnumpy(y_data[:,0]))
plt.plot(cp.asnumpy(Y[0,:]))


# In[ ]:


# OLS
q = cp.zeros([1,10])
v = cp.zeros([1,10])
X = cp.zeros([5*T,10])
for t in range(T):
    q = 5*t
    for i in range(5):
        v[:,(2*i):(2*i+2)] = cp.hstack([0*1,y_data[t,i]])
        X[q+i,:] = resh(v,[10])
        v = cp.zeros([1,10])


# In[ ]:


for i in range(5):
    v[:,(2*i):(2*i+2)] = cp.hstack([1,y_data[t,i]])
    X[q+i,:] = resh(v,[10])
    v = cp.zeros([1,10])


# In[ ]:


Y = resh(y_data,[133*5,1])


# In[ ]:


Beta = m(m(cp.linalg.inv(m(X.T,X)),X.T),Y)


# In[ ]:


y_hat = cp.zeros([5,T])
ERR = cp.zeros([5,T])
for t in range(T):
    y_hat[:,t] = resh(m(X[(5*t):(5*t+5),:],Beta),[5])
    ERR[:,t] = resh(y_hat[:,t],[5]) -resh(y_data[t,:],[5]) 


# In[ ]:


plt.plot(cp.asnumpy(y_hat[0,:]))


# In[ ]:


plt.plot(cp.asnumpy(y_data[:,0]))


# In[ ]:


cp.sum(ERR**2)


# In[ ]:





# In[ ]:




