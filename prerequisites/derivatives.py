from numpy import ndarray
import numpy as np
from typing import Tuple,Callable
import matplotlib.pyplot as plt

def plot_function(f:Callable[[ndarray],ndarray],plot_range:Tuple[float,float]):
    """plot the function f in the range given by plot_range"""
    start,end = plot_range
    x = np.linspace(start,end)
    y = f(x)
    plt.plot(x,y)
    plt.show()

def sigmoid(x:ndarray)->ndarray:
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
    plot_function(sigmoid,(-5.0,5.0))