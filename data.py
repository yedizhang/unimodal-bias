import numpy as np
from scipy.stats import invwishart
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'


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
        return gen_toy_data(args.rho, args.ratio, args.dataset_size, args.noise)
    if args.data == 'multi':
        return gen_multi_data(args.dataset_size, args.noise)
    elif args.data == 'xor':
        return gen_xor_data(args.var_lin, args.dataset_size)
    elif args.data == 'fission':
        return gen_fission_data(args.dataset_size)
    else:
        raise NotImplementedError


def gen_toy_data(rho, ratio, size, noise):
    """
    x1: shape (size, 1)
    x2: shape (size, 1)
    y: shape (size, 1)
    """
    mean = [0, 0]
    cov = [[1, rho*ratio],
           [rho*ratio, ratio**2]]
    pts = np.random.multivariate_normal(mean, cov, size)
    w = np.ones(2)
    y = pts @ w
    if noise != 0:
        y = y + np.random.normal(loc=0, scale=noise, size=size)
    print(pts.mean(axis=0), '\n', np.cov(pts.T), '\n', np.corrcoef(pts.T)[0, 1])
    return {"x1": pts[:, [0]],
            "x2": pts[:, [1]],
            "y": y[:, np.newaxis],
            "cov": cov,
            "w_gt": w}


def gen_multi_data(size, noise, dim=30):
    mean = np.zeros(dim)
    Psi = np.eye(dim)
    Psi[0:(dim//2)] = 2*Psi[0:(dim//2)]   # the mean of invwishart is Psi; we don't want cov=I when time ratio is trivially 1
    cov = invwishart.rvs(df=dim+2, scale=Psi)
    pts = np.random.multivariate_normal(mean, cov, size)
    w = np.ones(dim)
    y = pts @ w
    if noise != 0:
        y = y + np.random.normal(loc=0, scale=noise, size=size)
    return {"x1": pts[:, 0:(dim//2)],
            "x2": pts[:, (dim//2):dim],
            "y": y[:, np.newaxis],
            "cov": cov,
            "w_gt": w}


def gen_xor_data(var_lin, size):
    x1 = np.array([[1, 1],
                   [1, -1],
                   [-1, 1],
                   [-1, -1]])
    x1_xor = np.array([-1, 1, 1, -1])[:, np.newaxis]
    x1 = np.repeat(x1, size//4, axis=0)
    x1_xor = np.repeat(x1_xor, size//4, axis=0)
    x2 = np.random.normal(0, np.sqrt(var_lin), size)[:, np.newaxis]  # np.sqrt(np.sqrt(2)/2)
    y = x1_xor + x2
    return {"x1": x1,
            "x2": x2,
            "y": y}


def gen_fission_data(size):
    x = np.random.normal(0, 2, size)[:, np.newaxis]
    return {"x": x,
            "y1": 0.5*x,
            "y2": x}