import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = '16'


def gen_data(args):
    if args.data == 'toy':
        return gen_toy_data(noise=False, size=args.dataset_size)
        # vis_toy_data(x1, x2, y)
    elif args.data == 'xor':
        return gen_xor_data(var_lin=args.var_lin, size=args.dataset_size)
    else:
        return gen_data(args.data)


def gen_toy_data(noise=False, size=4096):
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
    return {"x1": x1,
            "x2": x2,
            "y": y}


def gen_multi_data(relation='redundancy', size=4096):
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

    return {"x1": x1,
            "x2": x2,
            "y": y}

def gen_xor_data(var_lin=1, size=4096):
    x1 = np.array([[1, 1],
                   [1, -1],
                   [-1, 1],
                   [-1, -1]])
    x1_xor = np.array([-1, 1, 1, -1])[:, np.newaxis]
    x1 = np.repeat(x1, size//4, axis=0)
    x1_xor = np.repeat(x1_xor, size//4, axis=0)

    # mean = [0, 0]
    # cov = [[1, 0],
    #        [0, 4]]
    # x2 = np.random.multivariate_normal(mean, cov, 4)
    # y = x1_xor + x2[:, 0] + x2[:, 1]

    x2 = np.random.normal(0, np.sqrt(var_lin), size)[:, np.newaxis]  # np.sqrt(np.sqrt(2)/2)
    y = x1_xor + x2

    return {"x1": x1,
            "x2": x2,
            "y": np.squeeze(y)}