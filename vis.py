import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
from util import *
plt.rc('axes', axisbelow=True)
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'
colors = [plt.get_cmap('Set1')(i) for i in range(9)]


def vis_toy_data(x1, x2, y, plot_2D=False):
    if plot_2D == True:
        plt.plot(x1, x2, '.', alpha=0.5)
        plt.axis('equal')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x1, x2, y, marker='o')
    plt.show()


def prep_axs(args):
    ax_num = 2
    ims = []
    fig = plt.figure(figsize=(5*ax_num, 5))
    fig.suptitle('{} net, XOR & Gaussian var={:.2f}'.format(args.mode, args.var_lin))
    ax1 = fig.add_subplot(1, ax_num, 1)
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([0, args.epoch])
    ax2 = fig.add_subplot(1, ax_num, 2)
    ax2.grid()
    ax2.axis('equal')
    return fig, [ax1, ax2], []


def vis_relu(args, data, W, losses, axs):
    losses = losses / losses[0] if len(losses) != 0 else losses
    im1, = axs[0].plot(losses, c='k', animated=True)
    if W.shape[-1] == 3:
        im2 = axs[1].scatter(W[:, 0], W[:, 1], c=W[:, 2], 
                        #   vmin=-1, vmax=1,
                        cmap='coolwarm', edgecolors='k', linewidths=0.25, s=10, animated=True)
    else:
        im2 = axs[1].scatter(W[:, 0], W[:, 1], c='k', linewidths=0.25, s=10, animated=True)
    return [im1, im2]


def plot_training(args, data, results):
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    losses, weights = results['Ls'], results['W']
    if args.mode == "deep_fusion":
        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.plot(losses / losses[0], linewidth=2, label="$L_f={}$".format(args.fuse_depth))
    else:
        plt.figure(figsize=(4, 3))
        if weights is not None:
            if args.data == 'multi':
                plt.plot(losses, linewidth=2, c='k', label="Loss")
                plt.plot(norm(weights[:, (args.in_dim//2):],axis=-1), linewidth=2, c='b', label=r"$||W_{A}^{tot}||$")
                plt.plot(norm(weights[:, 0:(args.in_dim//2)],axis=-1), linewidth=2, c='fuchsia', label=r"$||W_{B}^{tot}||$")
            else:
                plt.plot(losses / losses[0], linewidth=2, c='k', label="Loss")
                plt.plot(weights[:, -1]/weights[-1, -1], linewidth=2, c='b', label=r"$W_{A}^{tot}$")
                plt.plot(weights[:, 0]/weights[-1, 0], linewidth=2, c='fuchsia', label=r"$W_{B}^{tot}$")
    plt.title(args.mode)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim((0, args.epoch))
    plt.yticks([0,0.5,1])
    plt.tight_layout(pad=0.5)

    if args.plot_Eg:
        plt.figure(figsize=(4, 3))
        plt.plot(results['Ls'], linewidth=2, c='k', label="Loss")
        plt.plot(results['Eg'], linewidth=2, c='r', label="$E_g$")
        # plt.hlines(0.276, 0, args.epoch, 'grey', linestyles='dotted', linewidth=2)  # Eg_uni when dataset_size = 700
        # plt.yticks([0, 0.276, 1], [0, r'$E_g^{uni}$', 1])
        # plt.hlines(0.467, 0, args.epoch, 'grey', linestyles='dotted', linewidth=2)  # Eg_uni when dataset_size = 70
        # plt.yticks([0, 0.467, 1], [0, r'$E_g^{uni}$', 1])
        # plt.xticks([0, 173, 622, 1200], [0, '$t_1$', '$t_2$', 1200])  # late fusion early stopping times
        plt.title('{} training samples'.format(args.dataset_size))
        plt.xlabel("Epoch")
        plt.xlim((0, args.epoch))
        plt.legend(loc=1)
        plt.tight_layout(pad=0.5)