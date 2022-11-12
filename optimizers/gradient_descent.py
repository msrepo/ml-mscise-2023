from typing import Callable
import numpy as np
from numpy import ndarray
from functools import partial
import matplotlib.pyplot as plt

def quadratic(x,theta0,theta1, theta2):
    return theta0 + theta1 * x + theta2 * x ** 2

def df_dx_generic(f:Callable,x, delta_x = 0.0001):
    return (f(x+delta_x) - f(x - delta_x)) / (2 * delta_x)


def gradient_descent(cost_function:Callable,initial_guess, N_ITERS=100,lr=0.1,verbose=True):
    """_summary_

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
        guesses.append(initial_guess)
        if verbose:
            print(f'x {initial_guess:.4f} f(x) {sample_quadratic(initial_guess):.4f}')


    final_guess = guesses[-1]

    return final_guess, guesses

if __name__ == '__main__':
    # find the minima of this cost function
    sample_quadratic = partial(quadratic,theta0=1,theta1 = -3,theta2 = 2) # y = f(x) = 1 -3x + 2x^2

    final_guess,intermediate_guesses = gradient_descent(sample_quadratic,initial_guess=4.0,verbose=False)
    PLOT_RANGE = (-5,5,0.25)
    x = np.arange(*PLOT_RANGE)
    y = sample_quadratic(x)
    plt.plot(x,y)
    plt.scatter(intermediate_guesses,sample_quadratic(np.array(intermediate_guesses)),c='r')
    plt.show()
