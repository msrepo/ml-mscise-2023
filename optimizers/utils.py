import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from typing import Callable

def plot_3d(f: Callable,xlim=6.0,ylim=6.0,elev=40,azim=35,**kwargs):
    from matplotlib.colors import LogNorm
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x1 = np.linspace(-xlim,xlim,30)
    x2 = np.linspace(-ylim,ylim,30)
    X,Y = np.meshgrid(x1,x2)
    Z = f(X,Y)


    ax.plot_surface(X,Y,Z,cmap='jet',norm=LogNorm(),edgecolor='none',alpha=0.8,**kwargs)
    ax.view_init(elev,azim)
    return fig,ax


class TrajectoryAnimation3D(animation.FuncAnimation):
    def __init__(self, *paths, zpaths, labels=[], fig=None, ax=None, frames=None, 
                interval=60, repeat_delay=5, blit=True, **kwargs):
        self.fig = fig
        self.ax = ax
        
        self.paths = paths
        self.zpaths = zpaths
        
        if frames is None:
            frames = max(path.shape[1] for path in paths)
        from itertools import zip_longest

        self.lines = [ax.plot([], [], [], label=label, lw=2)[0] 
                    for _, label in zip_longest(paths, labels)]

        super(TrajectoryAnimation3D, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                frames=frames, interval=interval, blit=blit,
                                                repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return self.lines

    def animate(self, i):
        for line,path,zpath in zip(self.lines,self.paths,self.zpaths):
            line.set_data(path[0,:i], path[1,:i])
            line.set_3d_properties(zpath[:i])
        return self.lines