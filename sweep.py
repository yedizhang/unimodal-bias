import numpy as np
import matplotlib.pyplot as plt
from main import train
from data import gen_data
from util import *
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '12'
colors = [plt.get_cmap('Set1')(i) for i in range(9)]


def sweep(args):
    repeat = args.repeat
    lag, bias = 0, 0
    for _ in range(repeat):
        data = gen_data(args)
        results = train(data, args)
        losses, weights = results['Ls'], results['W']
        if losses[-1] < 1e-3:
            ta = time_half(args, weights[:, 1])
            tb = time_half(args, weights[:, 0], True)
            lag += tb / ta
            bias += (weights[(ta+tb)//2, 1] - weights[-1, 1]) / weights[-1, 1]
        else:
            repeat -= 1
    assert repeat != 0, "Warning: training did not converge!"
    return lag / repeat, bias / repeat


def toy_sweep(args):
    fig1, ax1 = plt.subplots(figsize=(4, 3.3))
    fig2, ax2 = plt.subplots(figsize=(4, 3.3))
    rho_theo, rho_exp = np.linspace(-0.92, 0.92, 100), np.linspace(-0.9, 0.9, 9)
    lag_lin, lag_relu = np.zeros(len(rho_exp)), np.zeros(len(rho_exp))
    bias_lin, bias_relu = np.zeros(len(rho_exp)), np.zeros(len(rho_exp))
    for k, ratio in enumerate([3, 2, 1.5, 1]):
        for i, rho in enumerate(rho_exp):
            args.rho, args.ratio = rho, ratio
            args.relu = 1  # linear
            lag_lin[i], bias_lin[i] = sweep(args)
            args.relu = 0  # ReLU
            lag_relu[i], bias_relu[i] = sweep(args)
        lag_theo = (ratio**2 - 1) / (1 - rho_theo**2) + 1
        ax1.plot(rho_theo, lag_theo, c=colors[k], label="$\sigma_A / \sigma_B = {}$".format(ratio))
        ax1.scatter(rho_exp, lag_lin, alpha=0.8, edgecolors=colors[k], facecolors='none', marker='o')
        ax1.scatter(rho_exp, lag_relu, alpha=0.8, c=colors[k], marker='x')
        if ratio != 1:
            ax2.plot(rho_theo, rho_theo/ratio, c=colors[k], label="$\sigma_A / \sigma_B = {}$".format(ratio))
            ax2.scatter(rho_exp, bias_lin, alpha=0.8, edgecolors=colors[k], facecolors='none', marker='o')
            ax2.scatter(rho_exp, bias_relu, alpha=0.8, c=colors[k], marker='x')
    ax1.set_xlabel(r"Correlation coefficient $\rho$")
    ax1.set_ylabel(r"Time ratio $t_B / t_A$")
    ax1.set_yscale('log')
    ax2.set_xlabel(r"Correlation coefficient $\rho$")
    ax2.set_ylabel(r"Misattribution $W_{A}^{tot}(\infty) - W_{A}^{tot}(t_{uni})$")
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper left')
    fig1.tight_layout(pad=0.2)
    fig2.tight_layout(pad=0.2)
    fig1.savefig("toy_sweep_time_{}hid_{}repeat.pdf".format(args.hid_width, args.repeat))
    fig2.savefig("toy_sweep_bias_{}hid_{}repeat.pdf".format(args.hid_width, args.repeat))
    plt.show()


def depth_single(args):
    assert args.mode == 'deep_fusion', "Cannot do deep_sweep for {} network".format(args.data)
    data = gen_data(args)
    plt.figure(figsize=(4, 3))
    for i in range(args.depth, 0, -1):
        args.fuse_depth = i
        train(data, args)
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig('depth{:d}_single.pdf'.format(args.depth))
    plt.show()


def rho_sweep(args):
    plt.figure(figsize=(4, 3))
    rho_theo = np.linspace(-0.92, 0.92, 100)
    rho_exp = np.linspace(-0.9, 0.9, 9)
    lag_exp, lag_theo = np.zeros(len(rho_exp)), np.zeros(len(rho_theo))
    for k, Lf in enumerate([4, 3, 2, 1]):
        for i, rho in enumerate(rho_exp):
            args.rho, args.fuse_depth = rho, Lf
            lag_exp[i], _ = sweep(args)
        for i, rho in enumerate(rho_theo):
            args.rho, args.fuse_depth = rho, Lf
            lag_theo[i] = lag_depth(args)
        plt.plot(rho_theo, lag_theo, c=colors[k], label="$L_f={}$".format(Lf))
        plt.scatter(rho_exp, lag_exp, alpha=0.8, edgecolors=colors[k], facecolors='none', marker='o')
        np.save('sweep_time_deep_Lf{}.npy'.format(Lf), [rho_exp, lag_exp])
    plt.xlabel(r"Correlation coefficient $\rho$")
    plt.ylabel(r"Time ratio $t_B / t_A$")
    plt.gca().set_yscale('log')
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("sweep_depth{}_sweep_ratio{}_{}hid_{}repeat.pdf".format(args.depth, args.ratio, args.hid_width, args.repeat))
    plt.show()
    

def ratio_sweep(args):
    ratio_theo, ratio_exp = np.linspace(1, 3, 100), np.linspace(1.2, 2.8, 5)
    lag_theo, lag_exp = np.zeros(len(ratio_theo)), np.zeros(len(ratio_exp))
    plt.figure(figsize=(4, 3))
    for k, Lf in enumerate([4, 3, 2, 1]):
        args.fuse_depth = Lf
        for i, ratio in enumerate(ratio_exp):
            args.ratio = ratio
            lag_exp[i], _ = sweep(args)
        for i, ratio in enumerate(ratio_theo):
            args.ratio = ratio
            lag_theo[i] = lag_depth(args)
        plt.plot(ratio_theo, lag_theo, c=colors[k], label='$L_f$={}'.format(args.fuse_depth))
        plt.scatter(ratio_exp, lag_exp, alpha=0.8, edgecolors=colors[k], facecolors='none', marker='o')
    plt.xlabel(r"Variance ratio $\sigma_A / \sigma_B = \sqrt {k}$")
    plt.ylabel(r"Time ratio $t_B / t_A$")
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("ratio_sweep.pdf")
    plt.show()


def init_sweep(args):
    init_theo, init_exp = np.linspace(0.01, 0.2, 100), np.linspace(0.03, 0.18, 5)
    lag_theo, lag_exp = np.zeros(len(init_theo)), np.zeros(len(init_exp))
    plt.figure(figsize=(4, 3))
    for k, Lf in enumerate([4, 3, 2, 1]):
        args.fuse_depth = Lf
        for i, init in enumerate(init_exp):
            args.init = init
            lag_exp[i], _ = sweep(args)
        for i, init in enumerate(init_theo):
            args.init = init
            if args.depth == args.fuse_depth:
                lag_theo = 4*np.ones(len(init_theo))
            else:
                lag_theo[i] = lag_depth(args)
        plt.plot(init_theo, lag_theo, c=colors[k], label='$L_f$={}'.format(args.fuse_depth))
        plt.scatter(init_exp, lag_exp, alpha=0.8, edgecolors=colors[k], facecolors='none', marker='o')
    plt.xlabel(r"Initialization $u_0$")
    plt.ylabel(r"Time ratio $t_B / t_A$")
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("init_sweep.pdf")
    plt.show()