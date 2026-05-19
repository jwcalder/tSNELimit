import numpy as np 
import matplotlib.pyplot as plt 
import plots
plt.ion()

n = 1000
lam = 200
x = np.linspace(0,1,n)
dx = x[1]-x[0]
u = np.ones(n)
u[:n>>1] = 0
u += np.random.randn(n)*0.1
u /= 50
v = u.copy()

fig = plt.figure()
ax = fig.add_subplot(111)
line2, = ax.plot(x, v, 'b-',label='t-SNE') # Returns a tuple of line objects, thus the comma
line1, = ax.plot(x, u, 'r-',label='Perona-Malik') # Returns a tuple of line objects, thus the comma
plt.ylim((np.min(u), np.max(u)))
plt.legend()


for i in range(1000):

    #Perona-Malik update
    GE = np.append(u[1:] - u[:-1],0)
    GW = np.insert(u[:-1] - u[1:],0,0)
    u += 0.25*(GE/(1 + GE**2/dx**2) + GW/(1 + GW**2/dx**2))/5

    #t-SNE update
    GE = np.append(v[1:] - v[:-1],0)
    GW = np.insert(v[:-1] - v[1:],0,0)
    PM = 0.5*(GE/(1 + GE**2/dx**2) + GW/(1 + GW**2/dx**2))

    #Repulsion part
    B = 1/(np.mean(dx/(np.abs(GE)+1e-10)))
    R = -0.5*lam*B*(GE/((np.abs(GE)+1e-5)**3/dx**3) + GW/((np.abs(GW)+1e-5)**3/dx**3))

    #Update
    v += (PM + R)/10

    #Update plot
    line2.set_ydata(v)
    line1.set_ydata(u)
    fig.canvas.draw()
    fig.canvas.flush_events()


