from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def quadratic(x,theta0,theta1, theta2):
    return theta0 + theta1 * x + theta2 * x ** 2

def quadratic_2d(x1,x2):
    return 3.7 + 2.0*x1 + 3.4*x2  + x1**2 + 1.3*x2**2

def df_wrt_x1_quadratic_2d(x1,x2):
    return 2.0 + 2.0*x1

def df_wrt_x2_quadratic_2d(x1,x2):
    return 3.4 + 2.6*x2

def bohachevsky(x1,x2):
    # https://www.sfu.ca/~ssurjano/boha.html
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7

def df_wrt_x1_bohachevsky(x1,x2):
    return 2*x1 + 3*np.pi*0.3*np.sin(3*np.pi*x1)

def df_wrt_x2_bohachevsky(x1,x2):
    return 4*x2 + 4*np.pi*0.4*np.sin(4*np.pi*x2)

def df_dx_generic(f:Callable,x, delta_x = 0.0001):
    return (f(x+delta_x) - f(x - delta_x)) / (2 * delta_x)


def gradient_descent(cost_function:Callable,initial_guess, N_ITERS=100,lr=0.1,verbose=True):
    """apply gradient descent to find the minima of the cost function

    Args:
        cost_function (Callable): function whose minima we want to find
        initial_guess (float): where do we want to start searching for minima?
        N_ITERS (int, optional): how many gradient descent steps to take. Defaults to 100.
        lr (float, optional): how big of a step do we want to take. Defaults to 0.1.
    """
    grad_fn = partial(df_dx_generic,cost_function,delta_x=0.0001)

    guesses = [initial_guess] 
    for i in range(N_ITERS):
        initial_guess = initial_guess - lr * grad_fn(initial_guess)
        guesses.append(initial_guess) # keep list of intermediate points for visualization later
        if verbose:
            print(f'x {initial_guess:.4f} f(x) {cost_function(initial_guess):.4f}')


    final_guess = guesses[-1]

    return final_guess, guesses

def gradient_descent_2d(cost_function: Callable,df_wrt_x1:Callable,df_wrt_x2:Callable,initial_guess,N_ITERS=100,lr=0.1,verbose=True):
    guesses = [initial_guess]
    for i in range(N_ITERS):
        x1, x2 = initial_guess
        initial_guess = initial_guess - lr * np.array([df_wrt_x1(x1,x2),df_wrt_x2(x1,x2)])
        guesses.append(initial_guess)
        if verbose:
            print(f'x {initial_guess} {cost_function(*initial_guess)}')
    final_guess = guesses[-1]
    return final_guess,guesses



if __name__ == '__main__':

    def quadratic_minimization_demo():
        # find the minima of this cost function
        sample_quadratic = partial(quadratic,theta0=1,theta1 = -3,theta2 = 2) # y = f(x) = 1 -3x + 2x^2

        final_guess,intermediate_guesses = gradient_descent(sample_quadratic,initial_guess=4.0,verbose=False)
        PLOT_RANGE = (-5,5,0.25)
        x = np.arange(*PLOT_RANGE)
        y = sample_quadratic(x)
        plt.plot(x,y)
        plt.scatter(intermediate_guesses,sample_quadratic(np.array(intermediate_guesses)),c='r')
        plt.show()
    
    from utils import plot_3d,TrajectoryAnimation3D


    final_guess,intermediate_guesses = gradient_descent_2d(quadratic_2d,df_wrt_x1_quadratic_2d,df_wrt_x2_quadratic_2d,np.array([-3.0,4.0]),lr=0.1,N_ITERS=1000)
    # plot function and animate descent steps
    minima = (-1.0, -3.4/2.6)
    fig,ax = plot_3d(quadratic_2d,azim=-50)
    ax.plot(*minima,quadratic_2d(*minima),'r*',markersize=10)
    paths = np.array(intermediate_guesses).T
    f = quadratic_2d  
    zpaths = [f(*path) for path in paths.T]
    anim = TrajectoryAnimation3D(paths,zpaths=[zpaths],fig=fig,ax=ax,frames=60)
    plt.show()