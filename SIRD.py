import numpy as np
import matplotlib.pyplot as plt


def SIRD_dynamics(t, y):
    
    b = .45
    gamma = 0.04
    mu = 0.01
    N = 1000.
    
    dSdt = -(b/N)*y[1]*y[0]
    dIdt = (b/N)*y[1]*y[0] - (gamma + mu)*y[1]
    dRdt = gamma*y[1]
    dDdt = mu*y[1]
    
    dy = np.array([dSdt, dIdt, dRdt, dDdt])
    
    return dy


def lorenz(t, ic):  

    x, y, z = ic
    b = 8./3.
    sigma = 10.
    r = 28.
    dXdt = sigma*(y - x)
    dYdt = r*x - y - x*z
    dZdt = x*y - b*z
    
    dy = np.array([dXdt, dYdt, dZdt])
    return dy


def FDWO(t, ic):
    x, y, z = ic
    f = 0.4
    d = .25
    dXdt = y
    dYdt = -d*y + x - x**3 + f*np.cos(z)
    dZdt = 1.
    
    dy = np.array([dXdt, dYdt, dZdt])
    return dy

def FDWO2(t, ic):
    x, y = ic
    f = 0.4
    d = .25
    dXdt = y
    dYdt = -d*y + x - x**3 + f*np.cos(t)
    dy = np.array([dXdt, dYdt])
    return dy


def odeRK4(func, tspan, IC):
    #declaring our variables and parameters
    t = tspan[0]
    h = np.diff(tspan)
    n = np.size(tspan)
    n2 = np.size(IC)
    
    
    #Declaring Our Matrices
    tval = np.zeros((n, 1))
    vals = np.zeros((n, n2))
    k = np.zeros((4, n2))
    y = np.zeros((1, n2))
    
    #Setting our initial conditions
    tval[0] = t
    vals[0, :] = IC

    for i in range(n-1):
        y[:] = func(t, vals[i, :])
        k[0, :] = np.inner(h[i], y[0, :])
        y[:] = func(t + h[i]/2, vals[i, :] + .5*k[0, :])
        k[1, :] = np.inner(h[i], y[0, :])
        y[:] = func(t + h[i]/2, vals[i, :] + .5*k[1, :])
        k[2, :] = np.inner(h[i], y[0, :])
        y[:] = func(t + h[i], vals[i, :] + k[2, :])
        k[3, :] = np.inner(h[i], y[0, :])
        
        #Setting Variables for Next Loop
        vals[i+1, :] = vals[i, :] + (1/6)*(k[0, :] + 2*k[1, :] + 2*k[2, :] + k[3, :])
        t = t + h[i]
        tval[i+1] = t

    return [tval, vals]

plt.figure(1)
tspan = np.arange(0., 100., 0.5)
IC = [995., 5., 0., 0.]
t, y = odeRK4(SIRD_dynamics, tspan, IC)
plt.plot(t, y[:, 0], 'r', t, y[:, 1], 'b', t, y[:, 2], 'g', t, y[:, 3], 'k')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Susceptible', 'Infected', 'Recovered', 'Fatalities'])
plt.title('Evolution of SIRD Dynamics')
plt.show()

plt.figure(2)
IC2 = [1., 1., 1.]
tspan2 = np.arange(0., 100., 0.01)
t, y = odeRK4(lorenz, tspan2, IC2)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(y[:, 0], y[:, 1], y[:, 2], label='Lorenz Attractor')
plt.show()

plt.figure(3)
tspan3 = np.arange(0, 100., 0.01)
IC3 = [0.1, 0.1, 0]
t, y = odeRK4(FDWO, tspan3, IC3)
IC4 = [0.1, 0.1]
t2, y2 = odeRK4(FDWO2, tspan3, IC4)
plt.plot(y[:, 0], y[:, 1], 'r--', y2[:, 0], y2[:, 1], 'b--', label='Forced Double Well Oscillator')
plt.show()