#!/usr/bin/env python
# coding: utf-8

# In[103]:


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
EY = cp.load('C:\Research\First Glance at Data\EY.npy')
EPI = cp.load('C:\Research\First Glance at Data\EPI.npy')
y_data = cp.load('C:\Research\SV\SVDATT.npy').T
EPEC= cp.vstack([EY[1:]/100-1,EPI[1:]-4])
y_data = y_data[:,0:216]
T = 216
J = 1000
m= cp.matmul
resh = cp.reshape
norm = cp.random.standard_normal


# In[104]:


y_data.shape


# In[105]:


#1 Interbank
#2 SPY
#3 10 yr
#4 US CPI
#5 Fed Funds
#6 GDP


# In[106]:


def propogate_EXPEC(l):
    
    y_dat = cp.zeros([T,4])
    y_dat[:,0] = y_data[:,5]
    y_dat[:,1] = y_data[:,3]
    y_dat[:,2] = y_data[:,4]
    y_dat[:,3] = y_data[:,1]
    
    t = l
    J = 10000
    
    phi = .01*cp.random.standard_normal([16,J])
    

    XXX = cp.zeros(T)
    
    EPEC_hat = cp.zeros([4,T])
    
    EE = cp.zeros([4,J])
    
    EPEC_hat = cp.zeros([4,T])

    while t <T:
        EY1 = cp.sum(phi_1[0:4,:]*cp.reshape(y_data1[t-1,0:3],[3,1]),0) 
        EY2 = cp.sum(phi_1[4:8,:]*cp.reshape(y_data1[t-1,0:3],[3,1]),0)
        EY3 = cp.sum(phi_1[8:12,:]*cp.reshape(y_data1[t-1,0:3],[3,1]),0)
        EY4 = cp.sum(phi_1[12:16,:]*cp.reshape(y_data1[t-1,0:3],[3,1]),0)
        EE = cp.reshape(y_dat[t,0:4],[4,1])
        EE1 = cp.vstack([EY1,EY2,EY3,EY4])
        EE_star = EE1
        for i in range(3):
            EY1 = cp.sum(phi_1[0:4,:]*EE[0:4,:],0)
            EY2 = cp.sum(phi_1[4:8,:]*EE[0:4,:],0)
            EY3 = cp.sum(phi_1[8:12,:]*EE[0:4,:],0)
            EY4 = cp.sum(phi_1[12:16,:]*EE[0:4,:],0)

            EE = EE1
            EE1= cp.vstack([EY1,EY2,EY3,EY4,EY5,EY6,EY7,EY8,EY9,EY10])
            
        errrs =    cp.sum(10*abs(EE1[0:2,:]-cp.reshape(EPEC[:,t],[2,1])),0)
        XXX[t] = cp.mean(errrs)/10
        w = (math.pi**-1)*cp.exp(-1/2*errrs)
        w = w/cp.sum(w)
        idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
        phi_1 = phi_1[:,idx]
        EE = EE[:,idx]
        EPEC_hat[0,t] = cp.mean(cp.reshape(w,[1,J])*EE1[0,:]) 
        EPEC_hat[1,t] = cp.mean(cp.reshape(w,[1,J])*EE1[1,:]) 
        PHI_1[:,t] = cp.mean(phi,1)
        # propogate phi
        err =(cp.reshape(y_dat[:,t],[4,1])-EE_star)
        err= cp.vstack([err,err,err,err,err])
        err = err*cp.vstack([resh(cp.array(list(y_dat[0,t])*4),[16,1]),resh(cp.array(list(y_dat[1,t])*4),[16,1]),resh(cp.array(list(y_dat[2,t])*4),[16,1]),resh(cp.array(list(y_dat[3,t])*4),[16,1])]) 
        phi = phi+.02*(err)[0:40,:] +.1*norm([40,J])
        
        t = t+1
    EX = cp.zeros([10,T])
    t = l
    while t<T:
        EX[:,t] = m(cp.kron(cp.identity(6),cp.reshape(y_dat[:,t],[1,4])),PHI_1[:,t])
        t = t+1
    #print(cp.sum(XXX))
    
    
    return EX


# In[143]:


y_dat[:,0,:].shape


# In[568]:


#1 Interbank
#2 SPY
#3 10 yr
#4 US CPI
#5 Fed Funds
#6 GDP
y_dat = cp.zeros([T,4,1])
# GDP
y_dat[:,0,:] = .000000001*resh(y_data[:,5],[216,1])
# PI
y_dat[:,1,:] = .000000001*resh(y_data[:,3],[216,1])
# FFR
y_dat[:,2,:] = .000000001*resh(y_data[:,4],[216,1])
# SPY
y_dat[:,3,:] = .000000001*resh(y_data[:,1],[216,1])

t = l
J = 10000

phi = .0001*cp.random.standard_normal([16,J])

PHI = cp.zeros([16,T])

XXX = cp.zeros(T)

EPEC_hat = cp.zeros([4,T])

EE = cp.zeros([4,J])

EPEC_hat = cp.zeros([4,T])

while t <T:
    EY1 = cp.sum(phi[0:4,:]*cp.reshape(y_dat[t-1,:],[4,1]),0) 
    EY2 = cp.sum(phi[4:8,:]*cp.reshape(y_dat[t-1,:],[4,1]),0)
    EY3 = cp.sum(phi[8:12,:]*cp.reshape(y_dat[t-1,:],[4,1]),0)
    EY4 = cp.sum(phi[12:16,:]*cp.reshape(y_dat[t-1,:],[4,1]),0)
    EE = cp.reshape(y_dat[t,0:4],[4,1])
    EE1 = cp.vstack([EY1,EY2,EY3,EY4])
    EE = EE1
    for i in range(2):
        EY1 = cp.sum(phi[0:4,:]*EE,0)
        EY2 = cp.sum(phi[4:8,:]*EE,0)
        EY3 = cp.sum(phi[8:12,:]*EE,0)
        EY4 = cp.sum(phi[12:16,:]*EE,0)
        
        EE1= cp.vstack([EY1,EY2,EY3,EY4])

        EE = EE1
    errrs =    cp.sum((EE1[0:2,:]-cp.reshape(EPEC[:,t],[2,1]))**2,0)
        
    XXX[t] = cp.mean(errrs)
    w = (math.pi**-1)*cp.exp(-1/2*errrs)
    w = w/cp.sum(w)
    idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
    phi = phi[:,idx]
    EE = EE[:,idx]
    EPEC_hat[0,t] = cp.sum(cp.reshape(w,[1,J])*EE1[0,:]) 
    EPEC_hat[1,t] = cp.sum(cp.reshape(w,[1,J])*EE1[1,:]) 
    EPEC_hat[2,t] = cp.sum(cp.reshape(w,[1,J])*EE1[2,:]) 
    EPEC_hat[3,t] = cp.sum(cp.reshape(w,[1,J])*EE1[3,:]) 
    PHI[:,t] = cp.sum(cp.reshape(w,[1,J])*phi,1)
    # propogate phi
    err =(cp.reshape(y_dat[t],[4,1])-EE1) 
    err = cp.vstack([err[0],err[0],err[0],err[0],err[1],err[1],err[1],err[1],err[2],err[2],err[2],err[2],err[3],err[3],err[3],err[3]])
    err = err*resh(cp.hstack([cp.array(list(y_dat[t-1,0])*4),cp.array(list(y_dat[t-1,1])*4),cp.array(list(y_dat[t-1,2])*4),cp.array(list(y_dat[t-1,3])*4)]),[16,1])
    phi = phi+.03*(delta) + .001*norm([16,J])

    
    
    t = t+1
EX = cp.zeros([T,4])
t = l
while t<T:
    EX[t,:] =resh(m(resh(PHI[:,t],[4,4]),resh(y_dat[t-1,:],[4,1])),[4])
    t = t+1
    
print(cp.mean(XXX))

plt.plot(cp.asnumpy(EPEC_hat[1,:]))


# In[569]:


t = l
Q1 = cp.zeros([T,1])
Q2 = cp.zeros([T,1])
Q3 = cp.zeros([T,1])
Q4 = cp.zeros([T,1])
EX = cp.zeros([T,4])

while  t < T-1:
    Q1[t]  = cp.sum(PHI[0:4,t]*cp.reshape(y_dat[t-1,:],[1,4]),1) 
    Q2[t] = cp.sum(PHI[4:8,t]*cp.reshape(y_dat[t-1,:],[1,4]),1)
    Q3[t] = cp.sum(PHI[8:12,t]*cp.reshape(y_dat[t-1,:],[1,4]),1)
    Q4[t] = cp.sum(phi[12:16,t]*cp.reshape(y_dat[t-1,:],[1,4]),1)
    EX[t,:] =resh(m(resh(PHI[:,t],[4,4]),resh(y_dat[t-1,:],[4,1])),[4])
    t = t+1


# In[235]:


plt.plot(cp.asnumpy(EPEC[1,:]))


# In[241]:


plt.plot(cp.asnumpy(EPEC[0,:]))


# In[575]:


EX.shape


# In[146]:


A,B,G,SIG_A,SIG_B,SIG_G = initialize(l)
t = l
P0  = 1*cp.random.standard_normal([6,6])
while t <T:
        A[:,t] = A[:,t-1]
        P = SIG_A+P0
        D = P +SIG[:,:,t-l]
        L = P
        R = cp.kron(cp.eye(6),cp.reshape(y_data[(t-l):(t),:],[1,l*6]))
        X = cp.kron(cp.identity(6),cp.reshape(EX[:,t],[1,6]))
        y_hat = A[:,t] + cp.matmul(R,B[:,t])+cp.matmul(X,G[:,t])
        A[:,t] = A[:,t]+cp.matmul((cp.matmul(L,cp.linalg.inv(D))),(cp.reshape(y_data[t,:],[6,1])-cp.reshape(y_hat,[6,1])))[:,0]      
        P0 = P - cp.matmul(L,cp.matmul(cp.linalg.inv(D),L.T))
        t = t+1


# In[580]:


# A block
def propogate_A(A,B,G,y_data,Sig_A,T,l,SIG,EX):
    t = l
    P0  = 1*cp.random.standard_normal([6,6])
    while t <T:
        A[:,t] = A[:,t-1]
        P = SIG_A+P0
        D = P +SIG[:,:,t-l]
        L = P
        R = cp.kron(cp.eye(6),cp.reshape(y_data[(t-l):(t),:],[1,l*6]))
        X = cp.kron(cp.identity(6),cp.reshape(EX[t,:],[1,4]))
        y_hat = A[:,t] + cp.matmul(R,B[:,t])+cp.matmul(X,G[:,t])
        A[:,t] = A[:,t]+cp.matmul((cp.matmul(L,cp.linalg.inv(D))),(cp.reshape(y_data[t,:],[6,1])-cp.reshape(y_hat,[6,1])))[:,0]      
        P0 = P - cp.matmul(L,cp.matmul(cp.linalg.inv(D),L.T))
        t = t+1
    return A


# In[581]:


# B block
def propogate_B(A,B,G,y_data,Sig_B,T,l,SIG,EX):
    t = l
    P0  = 1*cp.random.standard_normal([6*6*l,6*6*l])
    while t <T:
        B[:,t] = B[:,t-1]
        R = cp.kron(cp.eye(6),cp.reshape(y_data[(t-l):(t),:],[1,l*6]))        
        X = cp.kron(cp.identity(6),cp.reshape(EX[t,:],[1,4]))
        P = Sig_B+P0
        D = cp.matmul(cp.matmul(R,P),R.T) +SIG[:,:,t-l]
        L = cp.matmul(P,R.T)
        y_hat = A[:,t] + cp.matmul(R,B[:,t])+cp.matmul(X,G[:,t])
        B[:,t] = B[:,t] + cp.matmul((cp.matmul(L,cp.linalg.inv(D))),(cp.reshape(y_data[t,:],[6,1])-cp.reshape(y_hat,[6,1])))[:,0]     
        P0 = P - cp.matmul(L,cp.matmul(cp.linalg.inv(D),L.T))
        t = t+1
    return B


# In[602]:


# G block
def propogate_G(A,B,G,y_data,Sig_G,T,l,SIG,EX):
    ERR = cp.zeros([6,T])
    t = l
    P0  = 1*cp.random.standard_normal([6*4,6*4])
    while t <T:
        G[:,t] = G[:,t-1]
        R = cp.kron(cp.eye(6),cp.reshape(y_data[(t-l):(t),:],[1,l*6]))
        X = cp.kron(cp.identity(6),cp.reshape(EX[t,:],[1,4]))
        P =  Sig_G+P0
        D = cp.matmul(cp.matmul(X,P),X.T) +SIG[:,:,t-l]
        L = cp.matmul(P,X.T)
        y_hat = A[:,t] + cp.matmul(R,B[:,t])+cp.matmul(X,G[:,t])
        G[:,t] = G[:,t] + cp.matmul((cp.matmul(L,cp.linalg.inv(D))),(cp.reshape(y_data[t,:],[6,1])-cp.reshape(y_hat,[6,1])))[:,0]    
        P0 = P - cp.matmul(L,cp.matmul(cp.linalg.inv(D),L.T))
        t = t+1
    return G


# In[603]:


def initialize_SIG():
    SIG = cp.zeros([6,6,T])
    for j in range(6):
            SIG[j,j,:] = 1
    return SIG


# In[612]:


def Gen_likic1(y_data,A,B,G,EX,SIG,l):
    T = 216
    Y = cp.zeros([6,T])
    ERR = cp.zeros([T])
    error = cp.zeros([6,1,T])
    t = l
    while t<T:
        R = cp.kron(cp.eye(6),cp.reshape(y_data[(t-l):(t),:],[1,l*6]))
        X = cp.kron(cp.identity(6),cp.reshape(EX[t,:],[1,4]))
        Y[:,t] = A[:,t]  + cp.matmul(R,B[:,t])+cp.matmul(X,G[:,t])
        error[:,:,t] = cp.reshape((Y[:,t]- y_data[t,:].T),[6,1])
        X = cp.linalg.inv(SIG[:,:,t])
        ERR[t] = cp.float(cp.log((math.pi**-5)*abs(cp.linalg.det(SIG[:,:,t-l])**.5))+(-1/2)*cp.matmul(cp.matmul(error[:,:,t].T,X),error[:,:,t]))
        t = t+1 
    return cp.sum(ERR[l:T]),cp.reshape(error[:,:,l:T],[6,T-l]),cp.sum(error[:,:,l:T]**2)


# In[613]:


def initialize(l):
    A = .001*cp.random.standard_normal([6,T])
    B = .001*cp.random.standard_normal([6*6*l,T])
    G = .001*cp.random.standard_normal([6*4,T])
    Sig_A = .01*cp.diag(cp.random.standard_normal([6])**2)
    Sig_B =.01*(cp.diag(cp.random.standard_normal([6*6*l])**2))
    Sig_G = .01*(cp.diag(cp.random.standard_normal([6*4]))**2)
    return A,B,G,Sig_A,Sig_B,Sig_G


# In[639]:


def propogate_cov(SIG,ERR,l):
    J = 10000
    LOL = cp.zeros([T])
    d  = .01*cp.exp(cp.random.standard_normal([6,J]))
    h = .01*cp.random.standard_normal([15,J])
    H = cp.zeros([15,T-l])
    D = cp.zeros([6,T-l])
    x = cp.var(ERR[0,:],0)
    l = 15
    for t in range(T-l):
        eps = cp.random.standard_normal([6,J])
        zz = eps*d
        zz[0] =  zz[0]
        zz[1] =  zz[1]+ h[1,:]*eps[1,:]
        zz[2] =  zz[2]+ cp.sum(h[1:3,:]*eps[0:2,:],0)
        zz[3] =  zz[3]+ cp.sum(h[3:6,:]*eps[0:3,:],0)
        zz[4] =  zz[4]+ cp.sum(h[6:10,:]*eps[0:4,:],0)
        zz[5] =  zz[5]+ cp.sum(h[10:(15),:]*eps[0:5,:],0)
        Error= cp.sum((cp.reshape(ERR[:,t],[6,1]) - zz)**2,0);
        w =((1/(2*math.pi)**(-5)))*cp.exp(-(1/2)*Error)
        #w = Error;
        w[cp.where(cp.isnan(w))] =0 
        w = w/(cp.sum(w))
        idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
        h = h[:,idx]
        d = d[:,idx]
        d = cp.exp(d*.1*cp.random.standard_normal([6,J]))
        h = h*cp.random.standard_normal([15,J])
        H[:,t] = cp.mean(h,1)
        D[:,t] = cp.mean(d,1)

        SIG[:,:,t] = cp.vstack([cp.zeros([6]),
               cp.hstack([H[0:1,t],cp.zeros(5)]),
               cp.hstack([H[1:3,t],cp.zeros(4)]),
               cp.hstack([H[3:6,t],cp.zeros(3)]),
               cp.hstack([H[6:10,t],cp.zeros(2)]),
               cp.hstack([H[10:15,t],cp.zeros(1)])])+ cp.diag(D[:,t]**2) 
        
        SIG[:,:,t] = cp.tril(SIG[:,:,t]).T+cp.tril(SIG[:,:,t])-cp.diag(cp.diag(SIG[:,:,t])) 
        
    return  SIG


# In[640]:


def prop_cov(ndim,ZZ,para,l):
    V = cp.zeros([ndim,ndim])
    l = int(l)
    t=  l
    while t<T:
        V = cp.matmul(cp.reshape((ZZ[:,t]-ZZ[:,t-1]),[ndim,1]),cp.reshape((ZZ[:,t]-ZZ[:,t-1]),[1,ndim]))+V
        t = t+1
    
    V = V +.000000001*cp.eye(ndim)
    new = np.zeros((ndim,ndim))
    k = int(ndim*(ndim+1)/2)
    vals = np.random.standard_normal([k])
    inds = np.triu_indices_from(new)
    new[inds] = vals
    x = np.zeros(ndim)
    df = ndim+10
    for i in range(ndim):
        k = df+1-i
        x[i] = (np.random.chisquare(df = k))**.5    
    y = (new + np.diag(x)).T
    L = cp.linalg.cholesky(V)
    SIGMA = cp.matmul(cp.matmul(L.T,cp.asarray(y)),L)

    return  cp.linalg.inv(SIGMA + .0001*cp.ones([ndim,ndim])) 


# In[641]:


def draw_covariance(ZZ,ndim):
    iw = sc.stats._multivariate.wishart_gen()
    V = cp.zeros([ndim,ndim])
    t=  l
    while t<T:
        V = cp.matmul(cp.reshape((ZZ[:,t]-ZZ[:,t-1]),[ndim,1]),cp.reshape((ZZ[:,t]-ZZ[:,t-1]),[1,ndim]))+V
        t = t+1
    
    #V = cp.sum(V[:,:,l:],2)
    
    V = iw.rvs(ndim,cp.asnumpy(cp.linalg.inv(cp.eye(ndim)+V)))
    
    return V


# In[642]:


B.shape


# In[644]:


start = time.time()
l = 10
SIG = initialize_SIG()
A,B,G,SIG_A,SIG_B,SIG_G = initialize(l)
A = propogate_A(A,B,G,y_data,SIG_A,T,l,SIG,EX)
B = propogate_B(A,B,G,y_data,SIG_B,T,l,SIG,EX)
G = propogate_G(A,B,G,y_data,SIG_G,T,l,SIG,EX)
likic,ERR,err = Gen_likic1(y_data,A,B,G,EX,SIG,l)
print("likic before:", likic)
print("error before:", err)
SIG_B = cp.asarray(draw_covariance(B,6*6*l))
SIG_A = cp.asarray(draw_covariance(A,6))
SIG_G = cp.asarray(draw_covariance(G,6*4))
SIG = propogate_cov(SIG,ERR,l)
A = propogate_A(A,B,G,y_data,SIG_A,T,l,SIG,EX)
B = propogate_B(A,B,G,y_data,SIG_B,T,l,SIG,EX)
G = propogate_G(A,B,G,y_data,SIG_G,T,l,SIG,EX)
likic1, ERR1,err1 = Gen_likic1(y_data,A,B,G,EX,SIG,l)
print("Likic After:", likic1)
print("error After:", err1)
stop = time.time()
print("time:", stop-start)


# In[ ]:




