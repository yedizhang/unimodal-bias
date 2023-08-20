import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from config import config
from main import train
from data import gen_data
from util import *
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'


def toy_sweep_ratio(args, ratio, repeat=1):
    exp_list = np.linspace(0, 0.9, 10)
    lag_exp = np.zeros(len(exp_list))
    for _ in range(repeat):
        for i, rho in enumerate(exp_list):
            args.rho, args.ratio = rho, ratio
            data = gen_data(args)
            losses, weights = train(data, args)
            lag_exp[i] += time_half(weights[:, 0]) / time_half(weights[:, 1])
    lag_exp = lag_exp / repeat
    theo_list = np.linspace(0, 0.9, 50)
    lag_theo = 1 + (ratio**2 - 1) / (1 - theo_list**2)
    return exp_list, lag_exp, theo_list, lag_theo


def toy_sweep(args):
    plt.figure(figsize=(5, 4))
    plt.rc('axes', prop_cycle=(cycler('color', list(matplotlib.colors.BASE_COLORS))))
    repeat = 10
    for ratio in [3, 2, 1.5]:
        exp_list, lag_exp, theo_list, lag_theo = toy_sweep_ratio(args, ratio, repeat)
        plt.scatter(exp_list, lag_exp, label="$\sigma_A / \sigma_B = {}$".format(ratio))
        plt.plot(theo_list, lag_theo)
    plt.xlabel(r"Correlation coefficient $\rho$")
    plt.ylabel(r"Time lag $t_A / t_B$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_sweep_{}hid_{}repeat.svg".format(args.hid_width, repeat))
    plt.show()


def deep_sweep(args):
    assert args.data == 'deep_fusion', "Cannot do deep_sweep for {} datasets".format(args.data)
    data = gen_data(args)
    for i in range(1, args.depth+1):
        args.fuse_depth = i
        train(data, args)
    plt.figure(figsize=(10, 5))
    plt.ylabel("Loss")
    plt.title('Deep fusion with $L=${:d}, $L_f=${:d}, $\Sigma = diag[1,4]$'.format(args.depth, args.fuse_depth-1))
    plt.tight_layout()
    plt.savefig('depthfuse{:d}_fuse{:d}.svg'.format(args.depth, args.fuse_depth-1))
    plt.show()


def xor_sweep(args):
    assert args.data == 'xor', "Cannot do xor_sweep for {} datasets".format(args.data)
    vars = np.hstack((np.linspace(0.01, 2, 30),
                      np.linspace(2.1, 5, 10)))
    dirs = []
    for var in vars:
        args.var_lin = var
        x1, x2, y = gen_data(args)
        dir = train(x1, x2, y, args)
        dirs.append(dir)
        print(var)
    plt.scatter(vars, np.array(dirs), c='k')
    plt.xlabel("Variance of linear modality")
    plt.ylabel("Number of directions")
    plt.show()