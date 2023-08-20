import numpy as np


def sph2cart(polar, azimuth, r):
    polar = polar * np.pi / 180
    azimuth = azimuth * np.pi / 180
    x = r * np.sin(polar) * np.cos(azimuth)
    y = r * np.sin(polar) * np.sin(azimuth)
    z = r * np.cos(polar)
    return np.array([x, y, z])


def cart2sph(xyz):
    xy = xyz[:,0]**2 + xyz[:,1]**2
    sph = np.zeros(xyz.shape)
    sph[0] = np.sqrt(xy + xyz[:,2]**2)  # radius
    sph[1] = np.arctan2(np.sqrt(xy), xyz[:,2]) * 180 / np.pi  # polar, elevation angle defined from Z-axis down
    sph[2] = np.arctan2(xyz[:,1], xyz[:,0]) * 180 / np.pi  # azimuth
    return sph


def count_angle(sph):
    polar = (sph[:, 1]/4).astype(int) * 4
    azimuth = (sph[:, 2]/4).astype(int) * 4
    p_val, p_num = np.unique(polar, return_counts=True)
    a_val, a_num = np.unique(azimuth, return_counts=True)
    val, num = np.unique(polar*azimuth, return_counts=True)
    return len(num)


def time_half(arr):
    temp = (arr - 0.5) ** 2
    return np.argmin(temp)