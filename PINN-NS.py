# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:27:42 2022

@author: Riccardo
"""

import deepxde as dde
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

dde.config.set_random_seed(48)
dde.config.set_default_float('float64')

xmin, xmax = -4.0, 6.0
ymin, ymax = -4.0, 4.0

def boundaryNACA4D(M, P, SS, c, n):
    """
    Compute the coordinates of a NACA 4-digits airfoil
    
    Args:
        M:  maximum camber value (*100)
        P:  position of the maximum camber (*10)
        SS: maximum thickness (*100)
        c:  chord length
        n:  the total points sampled will be 2*n
    """
    m = M / 100
    p = P / 10
    t = SS / 100
    
    if (m == 0):
        p = 1
    
    # Chord discretization (cosine discretization)
    xv = np.linspace(0.0, c, n+1)
    xv = c / 2.0 * (1.0 - np.cos(np.pi * xv / c))
    
    # Thickness distribution
    ytfcn = lambda x: 5 * t * c * (0.2969 * (x / c)**0.5 - 
                                   0.1260 * (x / c) - 
                                   0.3516 * (x / c)**2 + 
                                   0.2843 * (x / c)**3 - 
                                   0.1015 * (x / c)**4)
    yt = ytfcn(xv)
    
    # Camber line
    yc = np.zeros(np.size(xv))
    
    for ii in range(n+1):
        if xv[ii] <= p * c:
            yc[ii] = c * (m / p**2 * (xv[ii] / c) * (2 * p - (xv[ii] / c)))
        else:
            yc[ii] = c * (m / (1 - p)**2 * (1 + (2 * p - (xv[ii] / c)) * (xv[ii] / c) - 2 * p))
    
    # Camber line slope
    dyc = np.zeros(np.size(xv))
    
    for ii in range(n+1):
        if xv[ii] <= p * c:
            dyc[ii] = m / p**2 * 2 * (p - xv[ii] / c)
        else:
            dyc[ii] = m / (1 - p)**2 * 2 * (p - xv[ii] / c)
            
    # Boundary coordinates and sorting        
    th = np.arctan2(dyc, 1)
    xU = xv - yt * np.sin(th)
    yU = yc + yt * np.cos(th)
    xL = xv + yt * np.sin(th)
    yL = yc - yt * np.cos(th)
    
    x = np.zeros(2 * n + 1)
    y = np.zeros(2 * n + 1)
    
    for ii in range(n):
        x[ii] = xL[n - ii]
        y[ii] = yL[n - ii]
        
    x[n : 2 * n + 1] = xU
    y[n : 2 * n + 1] = yU
    
    return np.vstack((x, y)).T


def navier_stokes(x, y): 
    """
    System of PDEs to be minimized: incompressible Navier-Stokes equations (V-P formulation)
    """
    Re = 50 #Reynolds number
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    u_x = dde.grad.jacobian(y, x, i = 0, j = 0)
    u_y = dde.grad.jacobian(y, x, i = 0, j = 1)
    
    v_x = dde.grad.jacobian(y, x, i = 1, j = 0)
    v_y = dde.grad.jacobian(y, x, i = 1, j = 1)
    
    u_xx = dde.grad.hessian(y, x, i = 0, j = 0, component = 0)
    u_yy = dde.grad.hessian(y, x, i = 1, j = 1, component = 0)
    
    v_xx = dde.grad.hessian(y, x, i = 0, j = 0, component = 1)
    v_yy = dde.grad.hessian(y, x, i = 1, j = 1, component = 1)
    
    p_x = dde.grad.jacobian(y, x, i = 2, j = 0)
    p_y = dde.grad.jacobian(y, x, i = 2, j = 1)
    
    continuity = u_x + v_y 
    momentum_x = u * u_x + v * u_y + p_x - 1.0 / Re * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - 1.0 / Re * (v_xx + v_yy)
    
    return [continuity, momentum_x, momentum_y]

# Geometry definition
farfield = dde.geometry.Rectangle([xmin, ymin], [xmax, ymax])
airfoil  = dde.geometry.Polygon(boundaryNACA4D(0, 0, 12, 1, 100)) 
geom     = dde.geometry.CSGDifference(farfield, airfoil)

# Boundaries definition
def boundary_farfield_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], xmin)

def boundary_farfield_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], ymax)

def boundary_farfield_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], xmax)

def boundary_farfield_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], ymin)

def boundary_airfoil(x, on_boundary):
    # Note: return on_boundary and airfoil.on_boundary(x) gives an error (Problem in dde Polygon method on_boundary?)
    return on_boundary and (not farfield.on_boundary(x))  

# Boundary values definition
def funU(x):
    return 1.0

def funV(x):
    return 0.0

#def funP(x):
#    return 0.0
 
# Boundary conditions assembly    
bc_inlet_u = dde.DirichletBC(geom, funU, boundary_farfield_inlet, component = 0)
bc_inlet_v = dde.DirichletBC(geom, funV, boundary_farfield_inlet, component = 1)

bc_top_u = dde.DirichletBC(geom, funU, boundary_farfield_top, component = 0)
bc_top_v = dde.DirichletBC(geom, funV, boundary_farfield_top, component = 1)

bc_bottom_u = dde.DirichletBC(geom, funU, boundary_farfield_bottom, component = 0)
bc_bottom_v = dde.DirichletBC(geom, funV, boundary_farfield_bottom, component = 1)

#bc_outlet_p = dde.DirichletBC(geom, funP, boundary_farfield_outlet, component = 2)
#bc_outlet_u = dde.NeumannBC(geom, funV, boundary_farfield_outlet, component = 0)

bc_airfoil_u = dde.DirichletBC(geom, funV, boundary_airfoil, component = 0)
bc_airfoil_v = dde.DirichletBC(geom, funV, boundary_airfoil, component = 1)

bcs = [bc_inlet_u, bc_inlet_v, bc_top_u, bc_top_v, bc_bottom_u, bc_bottom_v, bc_airfoil_u, bc_airfoil_v]

# Problem setup
data = dde.data.PDE(geom, navier_stokes, bcs, num_domain = 10000, num_boundary = 7500, num_test = 5000)

# Neural network definition
layer_size  = [2] + [50] * 6 + [3]
n           = 10
activation  = f"LAAF-{n} tanh" 
initializer = 'Glorot uniform'

net = dde.nn.FNN(layer_size, activation, initializer)

# Enforcing hard boundary conditions where easily possible 
def modify_output(X, Y):
    x, y     = X[:, 0:1], X[:, 1:2]
    u, v, p  = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
    u_new    = (x - xmin) * (y - ymin) * (y - ymax) * u / 100 + 1.0 #Note: divided by 100 to balance the effect of pre-multiplications
    v_new    = (x - xmin) * (y - ymin) * (y - ymax) * v / 20
    
    return tf.concat((u_new, v_new, p), axis=1)

net.apply_output_transform(modify_output)

# Model definition
model = dde.Model(data, net)
model.compile(optimizer = 'adam', lr = 1e-3, loss_weights = [1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10]) # Giving more weight to bcs (actually no needed for hard imposed ones)

#resampler      = dde.callbacks.PDEResidualResampler(period=1000)
#early_stopping = dde.callbacks.EarlyStopping(patience = 40000)

# Training strategy:
    # 5000 epochs with adam and lr = 1e-3
    # 5000 epochs with adam and lr = 1e-4
    # 10000 epochs with adam and lr = 1e-5
    # 10000 epochs with adam and lr = 1e-6
    # L-BFGS-B at the end to fine tuning the network parameters

losshistory, train_state = model.train(epochs = 5000, display_every = 100, model_save_path = './')
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

model.compile(optimizer = 'adam', lr = 1e-4, loss_weights = [1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10])
losshistory, train_state = model.train(epochs = 5000, display_every = 100, model_save_path = './')
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

model.compile(optimizer = 'adam', lr = 1e-5, loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
losshistory, train_state = model.train(epochs = 10000, display_every = 100, model_save_path = './')
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

model.compile(optimizer = 'adam', lr = 1e-6, loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
losshistory, train_state = model.train(epochs = 10000, display_every = 100, model_save_path = './')
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

model.compile(optimizer = 'L-BFGS-B', loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
losshistory, train_state = model.train(display_every = 100, model_save_path = './')
dde.saveplot(losshistory, train_state, issave = True, isplot = True)

# Plotting tool: thanks to @q769855234 code snippet
dx = 0.01
dy = 0.01
dt = 0.01
x = np.arange(xmin, xmax + dy, dx)
y = np.arange(ymin, ymax + dy, dy)

X = np.zeros((len(x)*len(y), 2))
xs = np.vstack((x,)*len(y)).reshape(-1)
ys = np.vstack((y,)*len(x)).T.reshape(-1)
X[:, 0] = xs
X[:, 1] = ys

# Model predictions generation
Y = model.predict(X)

u = Y[:, 0].reshape(len(y), len(x))
v = Y[:, 1].reshape(len(y), len(x))
p = Y[:, 2].reshape(len(y), len(x))

plt.figure(figsize = (16, 9))
plt.streamplot(x, y, u, v, density = 1.5)
plt.contourf(x, y, p)
plt.plot(boundaryNACA4D(0, 0, 12, 1, 100)[:, 0], boundaryNACA4D(0, 0, 12, 1, 100)[:, 1])
plt.colorbar()
plt.savefig('NACA0012NSp.png')

plt.figure(figsize = (16, 9))
plt.streamplot(x, y, u, v, density = 1.5)
plt.contourf(x, y, u)
plt.plot(boundaryNACA4D(0, 0, 12, 1, 100)[:, 0], boundaryNACA4D(0, 0, 12, 1, 100)[:, 1])
plt.colorbar()
plt.savefig('NACA0012NSu.png')

plt.figure(figsize = (16, 9))
plt.streamplot(x, y, u, v, density = 1.5)
plt.contourf(x, y, v)
plt.plot(boundaryNACA4D(0, 0, 12, 1, 100)[:, 0], boundaryNACA4D(0, 0, 12, 1, 100)[:, 1])
plt.colorbar()
plt.savefig('NACA0012NSv.png')
