from functools import partial
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

""" We can compute, both mathematically and in code, 
    the derivatives of nested/composite functions if the individual functions are [mostly] differentiable."""

def plot_function(f:Callable,plot_range:Tuple,show:bool=False,axes = None,title="",**kwargs):
    """plot the function f in the range given by plot_range.
    kwargs is used here to set linestyle of the plot"""
    start,end,stepsize = plot_range
    
    x = np.arange(start,end,stepsize)
    y = f(x=x)  # type: ignore
    if axes:
        axes.plot(x,y,**kwargs)
        axes.set_title(title)
    else:
        plt.plot(x,y,**kwargs)
        plt.title(title)
    if show:
        plt.show()

def linear(x,theta0,theta1):
    return theta0 + theta1*x

def quadratic(x,theta0,theta1, theta2):
    return theta0 + theta1 * x + theta2 * x ** 2

def sigmoid(x:ndarray)->ndarray:
    return 1.0 / (1 + np.exp(-x))

def relu(x: ndarray) -> ndarray:
    return x * (x > 0)

def df_dx_sigmoid(x: ndarray) -> ndarray:
    return sigmoid(x) * (1.0 - sigmoid(x))

def df_dx_relu(x: ndarray) -> ndarray:
    return 1.0 * (x > 0)

def df_dx_generic(f:Callable,x: ndarray, delta_x = 0.0001)->ndarray:
    return (f(x+delta_x) - f(x - delta_x)) / (2 * delta_x)

if __name__ == '__main__':
    PLOT_RANGE = (-5.0,5.0,0.25)
    fig = plt.figure()
    ax = fig.subplots(nrows=2,ncols=2)

    def plot_fx_dfdx(f: Callable,axes,title=''):
        plot_function(f,PLOT_RANGE,axes=axes)
        dfdx = partial(df_dx_generic,f=f)
        plot_function(dfdx,PLOT_RANGE,linestyle='dotted',axes=axes)
        
        # move title inside the plot
        plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
        plt.rcParams['axes.titlepad'] = -14  # pad is in points...
        axes.set_title(title)

    sample_linear = partial(linear,theta0=1,theta1=-3)
    plot_fx_dfdx(sample_linear,ax[0,0],title='y=1-3x')

    sample_quadratic = partial(quadratic,theta0=0,theta1=0,theta2=2)
    plot_fx_dfdx(sample_quadratic,ax[0,1],title='y=x^2')

    plot_fx_dfdx(relu,ax[1,0],title='Relu')
    plot_fx_dfdx(sigmoid,ax[1,1],title='sigmoid')

    plt.show()
