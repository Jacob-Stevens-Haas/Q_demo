# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:09:42 2018

@author: 600301
Version 0.3.0 - goes with Reinforcement Learning-v0.3.0.ipynb
"""

from scipy import integrate
import numpy as np
from itertools import product

def restart(mins, maxes):
    """
    provide a uniformly distributed initial state vector for training
    mins - minimum values for each state dimension
    maxes - maximum value for each state dimension
    """
    dims = mins.shape[0]
    return (maxes-mins) * np.random.rand(dims) + mins

def update_target(state, model, level = 1):
    """
    Find Q(s',a') - the predicted return for the predicted optimal next step.
    state - stacked vector of position and velocity coordinates of UUV, then target 
    model - model for prediction
    level - not in use.  Eventually, for refined grid search
    """
    dims = len(state)//4
    grid_pts = np.arange(-1, 2, 1)
    grid = [grid_pts for dim in range(0, dims)]
    mesh = [np.concatenate((state,np.array(ptuple))).reshape((1,-1))  for ptuple in product(*grid)]

    return np.array([model.predict(point) for point in mesh]).max()

def choose(state, model, eps, level = 1):
    """
    Determine the next action based on the current state
    Either randomly chooses a value, or seeks to find the argmax of a black-box function using meshes
    """
    dims = len(state)//4
    if np.random.rand()<eps:
        plas = [-1,0,1]
        pla_grid = [plas for k in range(0,dims)]
        pla_mesh = [np.array(element) for element in product(*pla_grid)]
        idx = np.random.choice(range(0,len(pla_mesh)))
        return pla_mesh[idx]
#         return np.random.rand(dims)

    grid_pts = np.arange(-1, 2, 1) # A single grid axis
    grid = [grid_pts for dim in range(0, dims)] # list of axes
#    product(*grid) produces an iterator of every point combination from each grid axis
    mesh = [np.concatenate((state,np.array(ptuple))).reshape((1,-1))  for ptuple in product(*grid)]

    surface = np.array([model.predict(point) for point in mesh])
    choice = surface.argmax()
    bestpwr = mesh[choice][0,dims*4:]
#     while level > 0: Make the mesh finer.  Not included in this iteration
    return bestpwr
    
def distance(positions):

    """
    positions - stacked vector of position and velocity coordinates of UUV, then target
    """
    dims = len(positions)//4
    x = positions[:dims]
    y = positions[dims*2:dims*3]
    
    return np.linalg.norm(y-x)

def step(state, action, dT, target_vel, limit = 100, mass=1, max_power=1, Cd=.2, rho=4.1):
    """
    state - stacked vector of position and velocity coordinates of UUV, then target
    action - vector of control PLAs
    dT - time step
    target_vel - velocity of target
    limit - pass done flag if any position coordinate exceets limit
    """
    t0 = 0
    dt = .01

    dims = len(state)//4
    xv0 = state[:dims*2]
    kset = action
    r_effect = .00
    r = integrate.ode(rhs).set_integrator('vode', method='adams')
    r.set_initial_value(xv0, t0).set_f_params(kset, r_effect, mass, max_power, Cd, rho)

    while r.successful() and r.t < dT:
        r.integrate(r.t+dt)
#         Now, store the intermediate value
#         ind = int(np.round(r.t/dt))
#         uvs[ind, :] = r.y
        
    uuv_pos = r.y
    target_pos = state[dims*2:dims*3]+dT*target_vel

    new_state = np.concatenate((uuv_pos, target_pos, target_vel))
    reward = -distance(new_state)
    
    done = any(abs(uuv_pos[:dims]) > limit) or any(abs(target_pos[:dims]) > limit)
    
    return new_state , reward, done

def rhs(t, xvs, ks, random_strength, mass, max_power, Cd, rho):
    """
    The right hand side function in  dx/dt = rhs(*args)
    This models position and velocity under constant power.  Motion opposed by drag
    
    t -   Time, but doesn't matter because equation is autonomous (unless we incorporate a forcing term in the future)
    xvs - Stacked vector of position and velocity coordinates
    ks -  Power level actuator percentage coordinates (one per dimension)
    random_strength - The variance of normal random effects on acceleration, relative to mass.  Can break ode solvers, so may need to set to 0
    mass, max_power - Self explanatory
    Cd, rho - drag coefficient and fluid density
    """
    dims = len(xvs)//2
    #Thrust - Drag
    #Note that derivative is infinite if v = 0, therefore, we keep the thrust denominator from being zero
    dx = xvs[dims:]
    rand_effect = np.array([   random_strength*np.random.randn()    for i in range(0,dims)])
    drag = np.array([   -np.sign(v)/(2*mass)*Cd*rho*v**2    for v in xvs[dims:]])
    thrust = np.array([   k*max_power/np.maximum(mass*abs(v), 1/(mass**2))    for (k,v) in zip(ks, xvs[dims:])])
    dv = rand_effect + thrust + drag
    return [*dx, *dv]
