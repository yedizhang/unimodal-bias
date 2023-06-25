import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from data import *
from util import *


def h_bluemean(m, sig):
    return - m + 2*m* stats.norm(0, sig**2).cdf(m) \
           + np.exp(- m**2 / (2*sig**2)) *2*sig / np.sqrt(2*np.pi)


def yh_mean(polar, azimuth, x, y, mode='early'):
    d = sph2cart(polar, azimuth, 1)
    if mode == 'early':
        h = np.maximum(x @ d, 0)
        y = np.repeat(y[:, np.newaxis], len(polar), axis=1)
        return np.mean(h*y, axis=0)
    elif mode == 'late':
        d[:2] = 0.5 * d[:2] / np.linalg.norm(d[:2])
        d[2] = 0.5
        h1 = x1 @ d[:2]
        h2 = np.squeeze(x2) * d[2]
        h = np.maximum(h1, 0) + np.maximum(h2, 0)
        return np.mean(y*h)


def cal_contour(data, mode='early', grid=100):
    x1, x2, y = data.values()
    x = np.concatenate((x1, x2), axis=1)
    angles = np.linspace(-180, 180, grid)
    polars, azimuths = np.meshgrid(angles, angles)
    polars, azimuths = polars.flatten(), azimuths.flatten()
    r = yh_mean(polars, azimuths, x, y, mode)
    xyz = sph2cart(polars, azimuths, np.abs(r)).T
    return xyz, r


def xor_contour(data, ax, mode='early', grid=100, surf=True, imshow=False):
    X, cov = cal_contour(data, mode, grid)
    if surf:
        X = X.reshape((grid, grid, 3))
        cov = cov.reshape((grid, grid, 1))
        surf = ax.plot_surface(X[:,:, 0], X[:,:, 1], X[:,:, 2],  \
                        facecolors=plt.cm.coolwarm(Normalize()(cov)), antialiased=False, shade=False)
    else:
        pts = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.array(cov), cmap='coolwarm')
        fig.colorbar(pts, orientation='vertical')
    ax.set_aspect('equal')
    ax.view_init(elev=20, azim=10, roll=0)
    if imshow:
        ax.set_title(mode + ' fusion, $\pm$1 XOR & Gaussian')
        ax.view_init(elev=90, azim=0, roll=0)
        plt.show()


if __name__ == "__main__":
    sig = np.sqrt(0.5)
    ms = np.linspace(-10, 10, 400)
    l = [h_bluemean(m, sig) for m in ms]
    plt.plot(ms, np.array(l), c='k')
    plt.plot(ms, h_bluemean(ms, sig), c='k')
    plt.show()

    angles = np.linspace(-180, 180, 200)
    vars = [1.5, 1, 0.5]
    for var in vars:
        x1, x2, y = gen_xor_data(var_lin=var)
        l = [yh_mean(p, 135, x1, x2, y, 'early') for p in angles]
        plt.plot(angles, np.array(l), label=var)
    plt.xlabel("polar angle")
    plt.ylabel("cov(y, h)")
    plt.title("Sectional view at azimuth=135")
    plt.legend()
    plt.show()

    data = gen_xor_data(1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xor_contour(data, ax, mode='early', surf=True, imshow=True)