import graphlearning as gl
import numpy as np 
import matplotlib.pyplot as plt
from tsne import tsne, tsne_torch
from scipy.spatial import distance
import plots
import utils
import sys
import torch
import os
import argparse

np.random.seed(1)

#Command line argument parsing
parser = argparse.ArgumentParser(description='t-SNE Continuum Limit Simulations')
parser.add_argument('--no_accel', action="store", dest='accel', default=False)
parser.add_argument('--init', action="store", dest='init', default='identity')
parser.add_argument('--c1', action="store", dest='c1', default=0.0)
parser.add_argument('--c2', action="store", dest='c2', default=0.0)
args = parser.parse_args()

#Parameters 
no_accel = args.accel  #Turn off GPU acceleration
use_accel = not no_accel and torch.accelerator.is_available()

#Initialization
init = args.init
#init = 'identity' #Initialize from continuum solution
#init = 'random' #Initialize at random

#Gaussian cluster means
#c1,c2 = -0.1,0.1
c1,c2 = float(args.c1),float(args.c2)

#Print out parameters
if use_accel:
    print("Using GPU acceleration.")
if init == 'random':
    print("Using random initialization.")
if init == 'identity':
    print("Using Continuum limit initialization.")
print("Gaussian means are (%.2f,%.2f)."%(c1,c2))

#Gaussian clusters
def rho(x):
    s = 1/np.sqrt(200)
    r = (0.4*np.exp(-100*(x - c1)**2) +  0.4*np.exp(-100*(x - c2)**2))/(s*np.sqrt(2*np.pi))
    return r + 0.1

#Support of eta should be [-1,1]
def eta(z):
    return np.maximum(3*(1 - z**2)/4,0)

#Definition of sigma for graph construction
def sigma(x):
    rhomax = np.max(rho(np.arange(-1,1,0.001)))#2.3567583341910257
    a = rhomax #Looks like this needs to be at least rhomax/2 for convergence
    return a/rho(x)

#Compute solution of Continuum Variational problem
u,hu,x,rhox,ux,T = utils.gen_solve(rho,eta,sigma,use_eta=True)

#Generate and save plots
plt.figure()
k = len(u)>>1
plt.plot(u[:k],hu[:k])
#plt.title('h(u)')
plots.savefig('figs/hu.pdf',axis=True,grid=True)

plt.figure()
plt.plot(x,rhox)
#plt.title('rho(x)')
plots.savefig('figs/rho_%.2f_%.2f.pdf'%(c1,c2),axis=True,grid=True)

plt.figure()
plt.plot(x,ux)
#plt.title("u = T'")
plots.savefig('figs/Tp_%.2f_%.2f.pdf'%(c1,c2),axis=True,grid=True)

plt.figure()
plt.plot(x,T)
#plt.title("T")
plots.savefig('figs/T_%.2f_%.2f.pdf'%(c1,c2),axis=True,grid=True)

#Generate Data to solve discrete equation
n = 2500
eps = 5/n #5 is heuristically the number of neighbors
X = utils.rejection_sample(n,rho)

#Construct epsilon-ball graph with sigma bandwidth
W = eta(distance.cdist(X,X,'euclidean')/(eps*sigma(X)))
W[range(n),range(n)]=0

#Run t-SNE or load if experiment saved
fname = 'data/tsne_%d_%s_%.2f_%.2f.npz'%(n,init,c1,c2)
if os.path.isfile(fname):
    M = np.load(fname)
    Y = M['Y']
    loss = M['loss']
else:
    Z = np.interp(X,x,T/eps) #To initialize from continuum solution
    Y,loss = tsne_torch(Z,W,h=n/2,num_iter=10000,dim=1,init=init,use_accel=use_accel)
    np.savez_compressed(fname,Y=Y,loss=loss)

#Save losses
np.savez_compressed('data/loss_%d_%s_%.2f_%.2f.npz'%(n,init,c1,c2),loss=loss)

#Check if Y flipped
if Y[0,:] > Y[-1,:]:
    Y = -Y

Y -= np.min(Y)
Y *= eps

#Plot map T
plt.figure()
plt.plot(X[:,0],Y[:,0],label='t-SNE')
plt.plot(x,T,linestyle='--',label='Continuum Limit')
plt.legend(loc='upper left')
plots.savefig('figs/tsne_exp_T_%d_%s_%.2f_%.2f.pdf'%(n,init,c1,c2),axis=True,grid=True)

#Plot derivative T'
plt.figure()
DX = X[1:,0] - X[:-1,0]
DY = Y[1:,0] - Y[:-1,0]
plt.plot(X[1:,0],DY/DX,label="t-SNE")
Tp = np.abs(T[1:] - T[:-1])/(x[1]-x[0])
plt.plot(x[1:],Tp,label="Continuum Limit")
plt.legend(loc='upper left')
plt.ylim((0,5))
plots.savefig('figs/tsne_exp_Tp_%d_%s_%.2f_%.2f.pdf'%(n,init,c1,c2),axis=True,grid=True)

