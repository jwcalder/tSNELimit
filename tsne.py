import numpy as np 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def perp(p):
    "Perplexity"

    p = p + 1e-10
    return 2**(-np.sum(p*np.log2(p),axis=1))

def pmatrix(X,sigma):
    "P matrix in t-SNE"

    n = len(sigma)
    I = np.zeros((n,n), dtype=int)+np.arange(n, dtype=int)
    dist = np.sum((X[I,:] - X[I.T,:])**2,axis=2)
    W = np.exp(-dist/(2*sigma[:,np.newaxis]**2))
    W[range(n),range(n)]=0
    deg = W@np.ones(n)
    return np.diag(1/deg)@W   #P matrix for t-SNE

def bisect(X,perplexity):
    "Bisection search to find sigma for a given perplexity"

    m = X.shape[0]
    sigma = np.ones(m)
    P = pmatrix(X,sigma)
    while np.min(perp(P)) < perplexity:
        sigma *= 2
        P = pmatrix(X,sigma)

    #bisection search
    sigma1 = np.zeros_like(sigma)
    sigma2 = sigma.copy()
    for i in range(20):
        sigma = (sigma1+sigma2)/2
        P = pmatrix(X,sigma)
        K = perp(P) > perplexity
        sigma2 = sigma*K + sigma2*(1-K)
        sigma1 = sigma1*K + sigma*(1-K)

    return sigma

def GL(W):
    "Returns Graph Laplacian for weight matrix W"
    deg = W@np.ones(W.shape[0])
    return np.diag(deg) - W


def tsne_torch(X,W,h=1,num_iter=1000,dim=2,init='random',use_accel=False):
    """t-SNE embedding

    Args:
        X: Data cloud
        W: Weight matrix
        h: Time step
        num_iter: Total number of iterations
        dim: dimension
        init: 'random', 'identity', or 'continuum'
        use_accel: Whether to use GPU

    Returns:
        Y: Embedded points
    """

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")


    n = X.shape[0]

    #Initialization
    if init == 'random':
        Y = np.random.rand(X.shape[0],dim)
    else:
        Y = X.copy()

    #Normalize by degree
    deg = W@np.ones(n)
    P = (1/n)*np.diag(1/deg)@W

    P = torch.from_numpy(P).float().to(device)
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)

    Y.requires_grad = True

    optimizer = optim.SGD([Y], lr=h)

    loss_vals = []
    #Main gradient descent loop
    for i in range(num_iter):
        optimizer.zero_grad()
        Q = torch.cdist(Y,Y,p=2.0)
        Q = Q**2
        A = torch.sum(P*torch.log(1 + Q))
        R = torch.log(torch.sum(1/(1 + Q))-n)
        loss = A + R
        loss_vals += [loss.detach().cpu().item()]
        loss.backward()
        optimizer.step()
        if i % int(num_iter/100) == 0:
            print(i,loss.detach().item(),A.detach().item(),R.detach().item())

    return Y.detach().cpu().numpy(), loss_vals


def tsne(X,perplexity=50,h=1,alpha=50,num_early=100,num_iter=1000,P=None,dim=2,init='random'):
    """t-SNE embedding

    Args:
        X: Data cloud
        perplexity: Perplexity (roughly how many neighbors to use)
        h: Time step
        alpha: Early exaggeration factor
        num_early: Number of early exaggeration steps
        num_iter: Total number of iterations
        P: Weight matrix
        dim: dimension
        init: 'random' or 'identity'

    Returns:
        Y: Embedded points
    """

    #Build graph using perplexity
    m = X.shape[0]
    if P is None:
        sigma = bisect(X,perplexity)
        P = pmatrix(X,sigma)
        P = (P.T + P)/(2*m)

    #For indexing
    I = np.zeros((m,m), dtype=int)+np.arange(m, dtype=int)

    #Initialization
    if init == 'random':
        Y = np.random.rand(X.shape[0],dim)
    elif init == 'identity':
        Y = X.copy()
    else:
        Y = init.copy()

    #Main gradient descent loop
    for i in range(num_iter):

        #Compute embedded matrix Q
        q = 1/(1+np.sum((Y[I,:] - Y[I.T,:])**2,axis=2))
        q[range(m),range(m)]=0
        Z = np.sum(q)
        Q = q/Z

        #Compute gradient
        if i < num_early: #Early exaggeration
            grad = 4*Z*(alpha*GL(P*Q) - GL(Q**2))@Y
        else:
            grad = 4*Z*GL((P-Q)*Q)@Y

        #Gradient descent
        Y -= h*grad
    
        #Percent complete
        if i % int(num_iter/100) == 0:
            print('%d%%'%(int(100*i/num_iter)))
            np.fill_diagonal(Q,1)
            print(np.sum(P*np.log(P/Q+1e-12)))

    return Y,P


