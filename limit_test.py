import graphlearning as gl
import numpy as np 
import matplotlib.pyplot as plt
from tsne import tsne, tsne_torch
from scipy.spatial import distance
import plots
import utils
import sys
import torch

#Parameters 

exit_early = False
no_accel = False
use_accel = not no_accel and torch.accelerator.is_available()
sort = False
init = 'identity'
init = 'random'

np.random.seed(1)

#Support of eta should be [-1,1]
def eta(z):
    return np.maximum(3*(1 - z**2)/4,0)

#Support of rho is [-1,1]
#One Cluster
#def rho(x,a=1.0):
#    return (a - x**2)/(2*(a-1/3))

#def rho(x):
#    return np.ones_like(x)/2

#One Cluster
#def rho(x):
#    return (1 - x**2)**2


#Two clusters
#def rho(x,a=1.0):
#    r1 = 6*np.maximum(0.25 - (x-0.5)**2,0)
#    r2 = 6*np.maximum(0.25 - (x+0.5)**2,0)
#    r3 = 3*(1 - x**2)/4
#    return 0.4*r1 + 0.4*r2 + 0.2*r3

#Gaussian clusters
def rho(x,a=1.0,c1=0.0,c2=0.0):
    s = 1/np.sqrt(200)
    r = (0.4*np.exp(-100*(x - c1)**2) +  0.4*np.exp(-100*(x - c2)**2))/(s*np.sqrt(2*np.pi))
    return r + 0.1

def sigma(x):
    rhomax = np.max(rho(np.arange(-1,1,0.001)))#2.3567583341910257
    a = rhomax #Looks like this needs to be at least rhomax/2 for convergence
    return a/rho(x)

ret_fam = False
if ret_fam:
    u,hu,x,rhox,ux,T,sols = utils.gen_solve(rho,eta,sigma,use_eta=True,return_family=ret_fam)
else:
    u,hu,x,rhox,ux,T = utils.gen_solve(rho,eta,sigma,use_eta=True,return_family=ret_fam)

plt.figure()
k = len(u)>>1
plt.plot(u[:k],hu[:k])
plots.savefig('hu.pdf',axis=True,grid=True)
plt.title('h(u)')

plt.figure()
plt.plot(x,rhox)
plt.title('rho(x)')
plt.savefig('rho.pdf')

plt.figure()
plt.plot(x,ux)
plt.title("u = T'")

plt.figure()
plt.plot(x,T)
plt.title("T")

if exit_early:
    sys.exit(1)

#Generate 1D data
n = 1000
#eps = 2*(1/n)**(1/4)
eps = 5/n #5 is heuristically the number of neighbors
X = utils.rejection_sample(n,rho)
#X = np.linspace(-1,1,n)[:,None]
#eps = 2*(X[1,0] - X[0,0])
plt.figure()
plt.hist(X,bins=50)
plt.savefig('tsne_real_data.pdf')


#epsilon-ball graph
W = eta(distance.cdist(X,X,'euclidean')/(eps*sigma(X)))
W[range(n),range(n)]=0

#Run t-SNE
Z = np.interp(X,x,T/eps) #To initialize from continuum solution
Y = tsne_torch(Z,W,h=n,num_iter=100000,dim=1,init=init,use_accel=use_accel)

#Check if Y flipped
if Y[0,:] > Y[-1,:]:
    Y = -Y

if sort:
    Y = utils.sort(Y)
Y -= np.min(Y)
Y *= eps
plt.figure()
plt.hist(Y,bins=50)
plt.savefig('tsne_real_data_Y.pdf')

#Plot map T
plt.figure()
if ret_fam:
    for Ts in sols:
        plt.plot(x,Ts,c='black',linewidth=0.1)
    
plt.plot(x,T,label='Minimizer of Continuum Equation')

plt.plot(X[:,0],Y[:,0],linewidth=1,label='t-SNE with n=%d points'%n)
plt.legend(loc='upper left')
plt.savefig('tsne_exp_%d_%.2f_%d_%s.pdf'%(n,eps,sort,init))

plt.figure()

DX = X[1:,0] - X[:-1,0]
DY = Y[1:,0] - Y[:-1,0]
plt.plot(X[1:,0],DY/DX,label="t-SNE |T'(x)| with n=%d points"%n)
Tp = np.abs(T[1:] - T[:-1])/(x[1]-x[0])
plt.plot(x[1:],Tp,label="Minimizer of Continuum Equation T'(x)")
plt.legend(loc='upper left')
plt.ylim((0,5))
plt.savefig('tsne_exp_Tp_%d_%.2f_%d_%s.pdf'%(n,eps,sort,init))

plt.figure()

plt.scatter(X[:,0],Y[:,0],s=0.2)
plt.savefig('tsne_points_%d_%.2f_%d_%s.pdf'%(n,eps,sort,init))


#plt.figure()
#epsy = 5*eps
#H = eta(distance.cdist(Y,Y,'euclidean')/epsy)
#rhoy = (1/(n*epsy**2)) * (H@np.ones(n))
#rhoX = rho(X).flatten()
#plt.plot(x[1:],Tp,label="Minimizer of Continuum Equation T'(x)")
#plt.plot(X,rhoX/rhoy,label="Kernel derivative of t-SNE")
#plt.legend()
#
#plt.figure()
#plt.plot(X,rhoy,label='Empirical rho_Y')
#plt.plot(x[1:],rhox[1:]/Tp,label="Continuum rho_Y")
#plt.legend()

plt.show()
