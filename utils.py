import numpy as np

def B(x,ux,rhox):
    dx = x[1]-x[0]
    return 1/np.sum(dx*(rhox**2/(ux+1e-10)))

#u can be a vector
def h(u,eta,dz=1e-6):
    z = np.arange(-1,1,dz)+dz/2
    u = np.reshape(u,(len(u),1))
    z = np.reshape(z,(1,len(z)))
    return 2*np.sum(eta(z)*u**3*z**2/(1+u**2*z**2),axis=1)*dz

def gen_solve(rho,eta,sigma,du=1e-6,delta=0.01,dx=0.001,return_all=True,tol=1e-12,use_eta=False,return_family=False):

    u = np.arange(0,10,du)

    if use_eta:
        print('Using precomputed h')
        u[0]=1
        hu = 3*(u+1/u) * (1 - np.arctan(u)/u) - u 
        hu[0]=0
        u[0]=0
    else:
        hu = h(u,eta)


    x = np.arange(-1,1,dx)
    rhox = rho(x)
    sigmax = sigma(x)
    ux = delta*np.ones_like(x)

    err = tol + 1
    bval = 0
    while err > tol:
        bval = B(x,ux/sigmax,rhox)
        rhs = bval*rhox*sigmax
        vx = np.interp(rhs,hu,u)
        err = np.max(np.absolute(ux-vx))/np.max(np.absolute(ux))
        ux = vx
    
    if use_eta:
        #ux[0]=1
        lhs = 3*(ux+1/ux) * (1 - np.arctan(ux)/ux) - ux 
        #lhs[0]=0
        #ux[0]=0
        rhs = B(x,ux/sigmax,rhox)*rhox*sigmax
        print('Error=',np.max(np.abs(lhs-rhs)))

    #Undo transformation
    ux = ux/sigmax

    #Integrate to get T
    T = np.cumsum(ux*dx)
    if return_family:
        bmin = bval/2
        bmax = 2*bval
        sols = [T]
        for b in np.linspace(bmin,bmax,20):
            rhs = b*rhox
            unew = np.interp(rhs,hu,u)
            Tnew = np.cumsum(unew*dx)
            sols += [Tnew]
        return u,hu,x,rhox,ux,T,sols
    else:
        if return_all:
            return u,hu,x,rhox,ux,T
        else:
            return T


def is_sorted(X):
    print(X.shape)
    return np.all(X[:-1,:] <= X[1:,:])

def sort(X):
    ind = np.argsort(X[:,0])
    X = X[ind,:]
    return X

#Domain of rho is assumed to be [-1,1]
def rejection_sample(n,rho,sort=True):
    X = 2*np.random.rand(10*n,1)-1
    Z = np.random.rand(10*n)
    rhox = rho(X[:,0])
    rhox = rhox/np.max(rhox)
    ind = rhox >= Z
    X = X[ind,:]
    X = X[:n,:]
    if sort:
        ind = np.argsort(X[:,0])
        X = X[ind,:]
    return X

def truncated_gaussian(n,b=2,sort=True):
    X = np.random.randn(3*n,1)
    ind = np.abs(X[:,0]) < b
    X = X[ind,:]
    X = X[:n,:]
    if sort:
        ind = np.argsort(X[:,0])
        X = X[ind,:]
    return X

