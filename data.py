import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = '16'


def prep_data(args, data, device):
    import torch
    if args.mode == "late_fusion" or args.mode == "deep_fusion":
        x1_tensor = torch.tensor(data['x1']).float().to(device)
        x2_tensor = torch.tensor(data['x2']).float().to(device)
        y_tensor = torch.tensor(data['y']).float().to(device)
        in_dim = [x1_tensor.size(-1), x2_tensor.size(-1)]
        out_dim = y_tensor.size(-1)
        return x1_tensor, x2_tensor, y_tensor, in_dim, out_dim
    elif args.data == 'fission':
        x_tensor = torch.tensor(data['x']).float().to(device)
        y = np.concatenate((data['y1'], data['y2']), -1)
        y_tensor = torch.tensor(y).float().to(device)
        in_dim = x_tensor.size(-1)
        if args.mode == 'fission':
            out_dim = [data['y1'].shape[-1], data['y2'].shape[-1]]
        else:
            out_dim = y_tensor.size(-1)
    else:
        x = np.concatenate((data['x1'], data['x2']), -1)
        x_tensor = torch.tensor(x).float().to(device)
        y_tensor = torch.tensor(data['y']).float().to(device)
        in_dim = x_tensor.size(-1)
        out_dim = y_tensor.size(-1)
    return x_tensor, y_tensor, in_dim, out_dim


def gen_data(args):
    if args.data == 'toy':
        # vis_toy_data(x1, x2, y)
        return gen_toy_data(noise=False, size=args.dataset_size)
    if args.data == 'multi':
        return gen_multi_data(args.dataset_size)
    elif args.data == 'xor':
        return gen_xor_data(var_lin=args.var_lin, size=args.dataset_size)
    elif args.data == 'fission':
        return gen_fission_data(args.dataset_size)
    else:
        raise NotImplementedError


def gen_toy_data(noise=False, size=4096):
    """
    x1: shape (size, 1)
    x2: shape (size, 1)
    y: shape (size, 1)
    """
    mean = [0, 0]
    cov = [[1, 0],
           [0, 4]]
    pts = np.random.multivariate_normal(mean, cov, size)
    x1 = pts[:, [0]]
    x2 = pts[:, [1]]
    y = x1 + x2
    if noise:
        n = np.random.normal(loc=0.0, scale=0.1, size=size)
        y = y + n
    print(pts.mean(axis=0), '\n', np.cov(pts.T), '\n', np.corrcoef(pts.T)[0, 1])
    return {"x1": x1,
            "x2": x2,
            "y": y}


def gen_multi_data(size=4096):
    mean = [0, 0, 0, 0]
    cov = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 4, 0],
           [0, 0, 0, 4]]
    pts = np.random.multivariate_normal(mean, cov, size)
    x1 = pts[:, 0:2]
    x2 = pts[:, 2:4]
    w1 = [1, 3]
    w2 = [2, 1]
    y = x1 @ w1 + x2 @ w2
    return {"x1": x1,
            "x2": x2,
            "y": y[:, np.newaxis]}


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
            "y": y}


def gen_fission_data(size=4096):
    x = np.random.normal(0, 2, size)[:, np.newaxis]
    return {"x": x,
            "y1": 0.5*x,
            "y2": x}