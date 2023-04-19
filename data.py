import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = '16'


def gen_toy_data(noise=False, size=5000):
    mean = [0, 0]
    cov = [[1, 0],
           [0, 4]]
    pts = np.random.multivariate_normal(mean, cov, size)
    x1 = pts[:, [0]]
    x2 = pts[:, [1]]
    y = pts[:, 0] + pts[:, 1]
    if noise:
        n = np.random.normal(loc=0.0, scale=0.1, size=size)
        y = y + n
    print(pts.mean(axis=0), '\n', np.cov(pts.T), '\n', np.corrcoef(pts.T)[0, 1])
    return x1, x2, y


def gen_data(relation='redundancy', size=5000):
    mean = [0, 0, 0, 0]
    cov = [[1, 0, 0, 0],
           [0, 4, 0, 0],
           [0, 0, 4, 0],
           [0, 0, 0, 1]]
    pts = np.random.multivariate_normal(mean, cov, size)
    x1 = pts[:, 0:3]
    x2 = np.concatenate((pts[:, 0:2], pts[:, [3]]), 1)
    if relation == 'redundancy':
        y = pts[:, 0] + pts[:, 1]
    elif relation == 'uniqueness':
        y = pts[:, 2]
    elif relation == 'synergy':
        y = pts[:, 2] + pts[:, 3]

    # mean = [0, 0, 0]
    # cov = [[1, 0, 0],
    #        [0, 1, 0],
    #        [0, 0, 4]]
    # pts = np.random.multivariate_normal(mean, cov, size)
    # x1 = pts[:, 0:2]
    # x2 = np.stack((pts[:, 0], pts[:, 2]), 1)
    # if relation == 'redundancy':
    #     y = pts[:, 0]
    # elif relation == 'uniqueness':
    #     y = pts[:, 1]
    # elif relation == 'synergy':
    #     y = pts[:, 1] + pts[:, 2]

    return x1, x2, y


def vis_toy_data(x1, x2, y, plot_2D=False):
    if plot_2D == True:
        plt.plot(x1, x2, '.', alpha=0.5)
        plt.axis('equal')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x1, x2, y, marker='o')
    plt.show()