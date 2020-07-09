# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:02:32 2018

@author: 600301
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as off
from itertools import count
from itertools import cycle

cmap = plt.get_cmap("tab10")

def plotPositionSlice(Xs, y, myfunc):
    """
    Returns a slice of the learned reward surface for UUV in position y with list (or numpy array) Xs.
    model - trained model
    
    """
    m = Xs.shape[0]
    velocities = Xs[:,1]
    predicted1 = [myfunc(np.array([[y,velocity,10,0,-1]])) for velocity in velocities]
    predicted1 = np.array(predicted1).reshape(1, m)
    plt.plot(velocities, predicted1.transpose(), label = 'Backwards')
    predicted2 = [myfunc(np.array([[y,velocity,10,0,0]])) for velocity in velocities]
    predicted2 = np.array(predicted2).reshape(1, m)
    plt.plot(velocities, predicted2.transpose(), label = 'Stationary')
    predicted3 = [myfunc(np.array([[y,velocity,10,0,1]])) for velocity in velocities]
    predicted3 = np.array(predicted3).reshape(1, m)
    plt.plot(velocities, predicted3.transpose(), label = 'Forwards')
    plt.xlabel('Velocity')
    plt.ylabel('Reward')
    plt.legend()
    return None
    
def plotVelocitySlice(x, Ys, myfunc):
    n = Ys.shape[1]
    posits = Ys[1,:]
    predicted1 = [myfunc(np.array([[posit,x,10,0,-1]])) for posit in posits]
    predicted1 = np.array(predicted1).reshape(1, n)
    plt.plot(posits, predicted1.transpose(), label = 'Backwards')
    predicted2 = [myfunc(np.array([[posit,x,10,0,0]])) for posit in posits]
    predicted2 = np.array(predicted2).reshape(1, n)
    plt.plot(posits, predicted2.transpose(), label = 'Stationary')
    predicted3 = [myfunc(np.array([[posit,x,10,0,1]])) for posit in posits]
    predicted3 = np.array(predicted3).reshape(1, n)
    plt.plot(posits, predicted3.transpose(), label = 'Forwards')
    plt.xlabel('Position')
    plt.ylabel('Reward')

    plt.legend()
    return None

def compareSlices(coordinates, point, predictor, immediate_reward, which = 0):
    fig = plt.figure(figsize = [8,4])
    if which == 0:
        fig.suptitle('The reward surface for UUV at velocity ' + str(point) + '.')
    else:
        fig.suptitle('The reward surface for UUV at position ' + str(point) + '.')
    fig.add_subplot(121)
    plt.title('Predicted Reward')
    if which == 0:
        plotVelocitySlice(point, coordinates, predictor)
    else:
        plotPositionSlice(coordinates, point, predictor)
#    plt.ylim(-10, 0)
    fig.add_subplot(122)
    # plotPositionSlice(Xs, 10, lambda x: stepPrint(x, step_simple, dT = 1))
    plt.title('Immediate Reward')
    if which == 0:
        plotVelocitySlice(point, coordinates, lambda x: stepPrint(x, immediate_reward, dT = 1))
    else:
        plotPositionSlice(coordinates, point, lambda x: stepPrint(x, immediate_reward, dT = 1))
    fig.subplots_adjust(wspace = .35)



def interactiveLearnedSurfaces(Xs, Ys, model, dimensions, control_options, fixed_coordinates,
                               control_names = None,
                               dimension_labels = None):
    
    """
    Create a 3D, interactive plot
    dimensions - tuple to determine dimensions of phase space (x and y axes) and dimension of control space (z axis)
                 phase space dimensions are in range(0,2*problem_dimensions), referring to the positions and velocities.
                 Control space dimensions are in range(0, problem dimensions)
    """
    m = Xs.shape[0]
    n = Ys.shape[1]
    ylist = Ys[1,:]
    xlist = Xs[:,1]
    data = []
    if control_names == None:
        control_names = ("Control " + str(i) for i in count(0))    
    if dimension_labels == None:
        dimension_labels = ["State Dim. "+str(dimensions[0]), "State Dim. "+str(dimensions[1])]    
    
    color0 = tuple(.999*np.array(cmap(0)[0:3]))
    
    for k, option in enumerate(control_options):
        mesh = [[np.concatenate((compose(x, y, dimensions, fixed_coordinates), 
                                          np.array([option])), axis = 1) for y in ylist] for x in xlist]
        mesh = np.array(mesh) #mesh.shape = (m,n,1, 2*problem_dimensions+1)
        predicted = np.apply_along_axis(modelApply1D, 3, mesh, model).reshape(m,n)
        
        if k ==0:
            colors = np.zeros(shape = predicted.shape)
            color1 = tuple(.999*np.array(cmap(1)[0:3]))
        else:
            colors = np.ones(shape = predicted.shape)
            color1 = tuple(.999*np.array(cmap(k)[0:3]))
            
        colorscale = [[0, 'rgb' + str(color0)], 
                      [1, 'rgb' + str(color1)]]
        trace_name =  next(control_names) if hasattr(control_names, "__next__") else control_names[k]
        
        trace = go.Surface(x = Xs, y = Ys, z = predicted,
                         surfacecolor=colors, 
                         opacity= 1.0, 
                         name=trace_name,
                         cmin=0,
                         cmax=1,
                         colorscale=colorscale,
                         showscale = False,
                         showlegend = True)
        data.append(trace)
 
    layout = go.Layout(
        title='The Neural Net\'s Learned Reward Function',
        showlegend = True,
        scene = dict(
            zaxis = dict(
                title = "Reward"
            ),
            yaxis = dict(
                title = dimension_labels[1]
            ),
            xaxis = dict(
                title = dimension_labels[0]
            ),
            aspectmode = 'cube'
        ), #Bonus material!!!
        margin = go.layout.Margin(
                l = 50,
                r = 50,
                b = 40,
                t = 20)
    )
    
    
    fig = go.Figure(data=data, layout=layout)
   
    off.iplot(fig)
    
    return None

    
def learnedSurface(Xs, Ys, model):
    m = Xs.shape[0]
    n = Ys.shape[1]
    posits = Ys[1,:]
    velocities = Xs[:,1]
    fig = plt.figure(figsize = [10,6])
    predicted1 = [[model.predict(np.array([[posit,velocity,10,0,-1]])) for posit in posits] for velocity in velocities]
    predicted1 = np.array(predicted1).reshape(m, n)
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xs, Ys, predicted1, label = '-1')
    predicted2 = [[model.predict(np.array([[posit,velocity,10,0,0]])) for posit in posits] for velocity in velocities]
    predicted2 = np.array(predicted2).reshape(m, n)
    ax.plot_surface(Xs, Ys, predicted2, label = '0')
    predicted3 = [[model.predict(np.array([[posit,velocity,10,0,1]])) for posit in posits] for velocity in velocities]
    predicted3 = np.array(predicted3).reshape(m, n)
    ax.plot_surface(Xs, Ys, predicted3, label = '1')
    plt.xlabel("Velocity", size = 'x-large')
    plt.ylabel("Position", size = 'x-large')
    ax.set_zlabel("Predicted Reward", size = 'x-large')

    fake2Dline1 = mpl.lines.Line2D([0],[0], linestyle="none", c=cmap(0), marker = 'o')
    fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="none", c=cmap(1), marker = 'o')
    fake2Dline3 = mpl.lines.Line2D([0],[0], linestyle="none", c=cmap(2), marker = 'o')
    ax.legend([fake2Dline1, fake2Dline2, fake2Dline3], ['Backwards', 'Stationary', 'Forwards'], numpoints = 1)
    # ax.legend()
    # ax.set_zlim(-10,2);
    fig.suptitle("The Neural Net's Learned Reward Function", weight = 'heavy', size = 'xx-large')
    fig.subplots_adjust(top = .98)
    return None

def restartPlot(inner_iters, max_iter, final_range):
    lns1 = plt.plot(inner_iters, label = "Inner Iterations")
    lns2 = plt.plot(max_iter*np.ones(inner_iters.shape[0]), "--", color = cmap(1), label = 'Maximum Iterations')
    plt.ylabel("Iteration")
    plt.twinx()
    lns3 = plt.plot(final_range, color = cmap(2), label = "Final Range")
    plt.ylabel("Range")
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)
    plt.xlabel('Trial')
    plt.title("Effective Learning: Inner Iterations Should Increase to Maximum\n Final Range Should Decrease to Zero")
    return None

def motionPlot(uuv_states, pwrs, total_iter, startiter=1, restarts=[], dim = 0):
    dims = uuv_states.shape[1]//2
    plt.figure(figsize = [12, 4])
    plt.title("UUV Explores State Space, Then Finds Target")
    plt.subplot(131)
    my_label_1 = 'coordinate ' + str(dim) +' pos'
    plt.plot(range(startiter,startiter+total_iter),uuv_states[:total_iter, dim], label = my_label_1)
    # plt.plot(range(startiter,startiter+total_iter),ys[:total_iter], label = 'y-coordinate')
    # plt.plot(range(startiter,startiter+total_iter),zs[:total_iter], label = 'z-coordinate')
    plt.plot(range(startiter,startiter+total_iter), 10*np.ones(total_iter), "--", label = 'Target')
    plt.xlabel("Iteration (time in seconds)")
    plt.title("Position")
    plt.ylabel("meters")
    plt.legend()
    for x in restarts:
        plt.axvspan(xmin=x, xmax=x+1, color = cmap(4))
    
    plt.subplot(132)
    my_label_2 = 'coordinate ' + str(dim) +' vel'
    plt.plot(range(startiter,startiter+total_iter),uuv_states[:total_iter, dims+dim], label = my_label_2)
    plt.xlabel("Iteration (time in seconds)")
    plt.title("Velocity")
    plt.ylabel("meters/second")
    for x in restarts:
        plt.axvspan(xmin=x, xmax=x+1, color = cmap(4))
        
    plt.subplot(133)
    plt.plot(range(startiter,startiter+total_iter), pwrs[0:total_iter, dim], label = my_label_1)
    plt.xlabel("Iteration (time in seconds)")
    plt.title("Power")
    plt.ylabel("+/- PLA")
    plt.subplots_adjust(wspace = .35)
    for x in restarts:
        plt.axvspan(xmin=x, xmax=x+1, color = cmap(4))
        
    return None
    
def learningPlot(predictions, targets, ranges, total_iter, startiter=1, restarts = [], N = 0):
    """
    Creates a plot of prediction error and immediate rewards for each iteration.  Over time, a good algorithm should 
    show decreasing error (it's learning) and increasing reward (It's choosing the right thing).
    predictions - The predictions for each step
    targets - the prediction target for each step
    ranges - the negative of range, i.e. the immediate reward for each step.
    total_iter, start_iter - self explanatory
    restarts - used to add restart bars to the plot
    N - smoothing window.  Use N=0 to not smooth, N = -1 to set window to a reasonable size based upon total iterations.
    """
    if N == -1:
        N = total_iter//10
    if N == 0:
        plt.plot(range(startiter, startiter+total_iter), predictions[0:total_iter]-targets[0:total_iter], label = "Predictor Error")
        plt.plot(range(startiter, startiter+total_iter), ranges[0:total_iter], label = "Immediate Rewards")
        plt.ylabel("Reward")
    else:
        smooth_ranges = np.convolve(ranges[0:total_iter], np.ones((N,))/N, mode='same')
        smooth_error = np.convolve(predictions[0:total_iter]-targets[0:total_iter], np.ones((N,))/N, mode = 'same')
        plt.plot(range(startiter, startiter+total_iter), smooth_error, label = "Predictor Error")
        plt.plot(range(startiter, startiter+total_iter), smooth_ranges, label = "Immediate Rewards")
        plt.ylabel("Moving Average Reward")

    plt.title("Effective Learning: Reward Should Increase\n Prediction Error Should Decrease")
    plt.xlabel("Iteration")
    plt.legend()
    return None

def runningAverage(myVec):
    
    return myVec
    
def splitState(state, k=1):
    """
    splits a numpy array or list for passing into step function 
    state - 1 x n, 2D numpy array
    k - desired number of entries in second return item
    """
    return state[0:-k],[state[-1: state.shape[0]]]

def stepPrint(myvar, stepfunc, dT=1, test_tgt_vel=np.array([0]), limit=30, mass=10, pmax=10, Cd = .1, rho = 4.1):
    """
    Used for handling the step function when input is 2d numpy array including state and action
    (just as it would be passed to model.predict)
    myvar - current state concatenated with action in 2d, 1xn numpy array
    stepfunc - function to step with    
    """
    return stepfunc(*splitState(myvar[0,:], 1), dT, test_tgt_vel, limit = 30,
                    mass = mass, max_power=pmax, Cd=Cd, rho=rho)[1]

def singleSurface(Xs, Ys,action, predictor, myname = None, fig = plt.figure(), **kwargs):
    """
    DEPRECATED
    Old way of plotting a surface using matplotlib
    """
    m, n = Xs.shape
    xlist = Xs[:,1]
    ylist = Ys[1,:]
    ax = fig.gca(projection = '3d')
    predicted = [[predictor(np.array([[y,x,10,0,action]])) for y in ylist] for x in xlist]
    predicted = np.array(predicted).reshape(m, n)
    ax.plot_surface(Xs, Ys, predicted, **kwargs)
    plt.xlabel("Velocity")
    plt.ylabel("Position")
    if myname:
        ax.set_zlabel(str(myname) + " Reward")
        
    return None

def compose(x, y, dimensions, fixed_coordinates):
    """
    Returns a complete state array ([UUV positions, UUV velocities, Target positions, Target velocities]) by inserting
    x and y into the fixed coordinates at the places defined by dimensions.
    x, y - two floats
    dimensions - 2-tuple of where to insert x and y.  Each entry must be in range(0, len(fixed_coordinates)+2).
    fixed_coordinates - numpy array of coordinates
    """
    low = min(*dimensions)
    high = max(*dimensions)
    full_coordinates = np.zeros((1, len(fixed_coordinates)+2))
    full_coordinates[0,dimensions[0]] = x
    full_coordinates[0,dimensions[1]] = y
    full_coordinates[0,0:low] = fixed_coordinates[0:low]
    full_coordinates[0,low+1:high] = fixed_coordinates[low:high-1]
    full_coordinates[0,high+1:len(fixed_coordinates)+2] = fixed_coordinates[high-1:len(fixed_coordinates)]
    return full_coordinates

def modelApply1D(myArray, model):
    """
    Apply model.predict to a 1D array myArray, even though model.predict takes a 1xn array.  
    This is an in intermediate step since numpy.apply_along_axis requires the input be 1d.
    """
    return model.predict(myArray.reshape((1,-1))) 