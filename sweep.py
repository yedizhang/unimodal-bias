import numpy as np
from scipy.integrate import quad
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from config import config
from main import train
from data import gen_data
from util import *
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '12'
colors = [plt.get_cmap('Set1')(i) for i in range(9)]


def integrand(u, wa, wb, L, Lf, init):
    k = wb/wa
    if Lf == 2:
         ub = np.exp(k*np.log(u) + (1-k)*np.log(init))
    else:
        ub = (k*u**(2-Lf) + (1-k)*init**(2-Lf)) ** (1/(2-Lf))
    denominator = wa * u**(Lf-1) * (u**2 + ub**2) ** ((L-Lf)/2)
    return 1/denominator


def lag_depth(args):
    if args.fuse_depth == 1:
        return 1
    else:
        wa = args.ratio**2 + args.rho * args.ratio
        wb = 1 + args.rho * args.ratio
        L, Lf = args.depth, args.fuse_depth
        ua0 = args.init
        I = quad(integrand, ua0, 1, args=(wa, wb, L, Lf, ua0))
        if Lf == 2:
            ln_ub0 = np.log(1/args.init) * (1-wb/wa)
            lag = ln_ub0 * wa**(Lf/L-1) / (1-args.rho**2)
        else:
            ub0 = args.init * (1-wb/wa) ** (1/(2-Lf))
            lag = ub0**(2-Lf) * args.ratio**(Lf/L-1) / ((Lf-2) * (1-args.rho**2))
        return 1 + lag/I[0]


def sweep(args):
    repeat = args.repeat
    lag, bias = 0, 0
    for _ in range(repeat):
        data = gen_data(args)
        losses, weights = train(data, args)
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
            args.activation = 'linear'
            lag_lin[i], bias_lin[i] = sweep(args)
            args.activation = 'relu'
            lag_relu[i], bias_relu[i] = sweep(args)
        lag_theo = (ratio**2 - 1) / (1 - rho_theo**2) + 1
        ax1.plot(rho_theo, lag_theo, c=colors[k], label="$\sigma_A / \sigma_B = {}$".format(ratio))
        ax1.scatter(rho_exp, lag_lin, alpha=0.8, edgecolors=colors[k], facecolors='none', marker='o')
        ax1.scatter(rho_exp, lag_relu, alpha=0.8, c=colors[k], marker='x')
        np.save('sweep/time_late_ratio{}.npy'.format(ratio), [rho_exp, lag_lin, lag_relu])
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
    fig1.savefig("sweep/toy_sweep_time_{}hid_{}repeat.pdf".format(args.hid_width, args.repeat))
    fig2.savefig("sweep/toy_sweep_bias_{}hid_{}repeat.pdf".format(args.hid_width, args.repeat))
    plt.show()


def depth_sweep(args):
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
        np.save('sweep/time_deep_Lf{}.npy'.format(Lf), [rho_exp, lag_exp])
    plt.xlabel(r"Correlation coefficient $\rho$")
    plt.ylabel(r"Time ratio $t_B / t_A$")
    plt.ylim((0.86, 24.12))
    plt.gca().set_yscale('log')
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("sweep/depth{}_sweep_ratio{}_{}hid_{}repeat.pdf".format(args.depth, args.ratio, args.hid_width, args.repeat))
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
    plt.savefig('depthfuse{:d}.pdf'.format(args.depth))
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