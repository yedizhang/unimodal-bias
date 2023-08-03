import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
from numerical import *
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'


def vis_toy_data(x1, x2, y, plot_2D=False):
    if plot_2D == True:
        plt.plot(x1, x2, '.', alpha=0.5)
        plt.axis('equal')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x1, x2, y, marker='o')
    plt.show()


def xor_sweep(args):
    assert args.data == 'xor', "Cannot do xor_sweep for {} datasets".format(args.data)
    from data import gen_data
    from main import train
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


def vis_relu_3d(W):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(W[:, 0], W[:, 1], W[:, 2], 
               c=W[:, 2], cmap='coolwarm', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('xor 1')
    ax.set_ylabel('xor 2')
    ax.set_title('Late fusion ReLU net, +-1XOR & Gaussian')
    ax.set_box_aspect([1,1,1])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def prep_axs(args):
    ax_num = 2+args.vis_contour
    if args.vis_feat:
        ims = []
        fig = plt.figure(figsize=(5*ax_num, 5))
        fig.suptitle('{} {} net, XOR & Gaussian var={:.2f}'.format(args.mode, args.activation, args.var_lin))
        ax1 = fig.add_subplot(1, ax_num, 1)
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlim([0, args.epoch])
        ax2 = fig.add_subplot(1, ax_num, 2)
        ax2.grid()
        ax2.axis('equal')
        return fig, [ax1, ax2], []
    if args.vis_contour:
        ax3 = fig.add_subplot(1,ax_num,3, projection='3d')
        plt.tight_layout()
        return fig, [ax1, ax2, ax3], []


def vis_relu(args, data, W, losses, axs):
    losses = losses / losses[0] if len(losses) != 0 else losses
    im1, = axs[0].plot(losses, c='k', animated=True)
    if W.shape[-1] == 3:
        im2 = axs[1].scatter(W[:, 0], W[:, 1], c=W[:, 2], 
                        #   vmin=-1, vmax=1,
                        cmap='coolwarm', edgecolors='k', linewidths=0.25, s=10, animated=True)
    else:
        im2 = axs[1].scatter(W[:, 0], W[:, 1], c='k', linewidths=0.25, s=10, animated=True)
    if len(axs) == 3:
        im3 = xor_contour(data, axs[2], args.mode)
        return [im1, im2, im3]
    else:
        return [im1, im2]


def plot_training(args, losses, weights=None):
    if args.mode == "deep_fusion":
        plt.rc('axes', prop_cycle=(cycler('color', list(matplotlib.colors.BASE_COLORS))))
        plt.plot(losses / losses[0], alpha=0.7, label=args.fuse_depth-1)  # label=args.fuse_depth-1, str(args.hid_width)+" hidden neurons"
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(losses / losses[0], c='k', alpha=0.7, label="loss")
    plt.xlabel("Epoch")
    # plt.ylim((-0.05, 1.05))
    plt.title(args.mode)  # args.mode+" linear network with 8 hidden neurons"
    if weights is not None:
        plt.plot(weights[:, 0], color='g', alpha=0.7, label="$W_{1A}$")
        plt.plot(weights[:, 1], color='m', alpha=0.7, label="$W_{1B}$")
        # if args.mode == "shallow":
        #     t = np.arange(args.epoch)
        #     tau = 0.5 / args.lr
        #     w1 = - np.exp(-2*t/tau) + 1
        #     w2 = - np.exp(-4*t/tau) + 1
        # if args.mode == "late_fusion":
        #     t = np.arange(args.epoch)
        #     tau = 0.5 / args.lr
        #     c = 1 / args.init ** 2
        #     w1 = 1 / (c * np.exp(-2*t/tau) + 1)
        #     w2 = 1 / (c * np.exp(-8*t/tau) + 1)
        # plt.plot(w1, 'g--')
        # plt.plot(w2, 'm--')
    plt.legend()