import numpy as np
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler
from util import *
from numerical import *
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
        fig.suptitle('{} net, XOR & Gaussian var={:.2f}'.format(args.mode, args.var_lin))
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


def fig_feat(args, W, t):
    x = np.arange(-0.2, 0.205, 0.005)
    y = np.arange(-0.2, 0.205, 0.005)
    X, Y = np.meshgrid(x, y)
    Z = bivariate_normal(X, Y, 2, 1)

    plt.figure(figsize=(4.2, 3))
    if args.relu == 0:
        plt.scatter(W[:, 0], W[:, 1], linewidths=0.25, s=10, c=W[:, 2], 
                        #   vmin=-1, vmax=1,
                        cmap='coolwarm', edgecolors='k')
        cbar = plt.colorbar(fraction=0.046, pad=0.08)
        cbar.set_label('$W^1_A$')
        plt.xlabel("$W^1_{B1}$")
        plt.ylabel("$W^1_{B2}$")
        plt.xlim((-0.3, 0.3))
        plt.ylim((-0.3, 0.3))
    else:
        plt.scatter(W[:, -1], W[:, 0], linewidths=0.25, s=10, c='k')
        plt.contourf(X, Y, Z, 30, vmin=np.min(Z)-0.001, cmap=plt.cm.bone)
        plt.annotate("", xy=(0.1, 0.1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='r', mutation_scale=20))
        plt.annotate("", xy=(0.137, 0.034), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='lime', mutation_scale=20))
        plt.xlabel("$W^1_A$")
        plt.ylabel("$W^1_B$")
        plt.xticks([-0.1, 0, 0.1])
        plt.yticks([-0.1, 0, 0.1])
    plt.axis('equal')
    plt.gca().set_adjustable("box")
    plt.title("Epoch={}".format(t), fontsize=14)
    plt.tight_layout(pad=0.2)
    plt.show()


def plot_training(args, data, results):
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    losses, weights = results['Ls'], results['W']
    if args.mode == "deep_fusion":
        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.plot(losses / losses[0], linewidth=2, label="$L_f={}$".format(args.fuse_depth))  # label=args.fuse_depth-1, str(args.hid_width)+" hidden neurons"
    else:
        plt.figure(figsize=(4, 3))
        if weights is not None:
            if args.data == 'multi':
                plt.plot(losses, linewidth=2, c='k', label="Loss")
                plt.plot(norm(weights[:, (args.in_dim//2):],axis=-1), linewidth=2, c='b', label=r"$||W_{A}^{tot}||$")
                plt.plot(norm(weights[:, 0:(args.in_dim//2)],axis=-1), linewidth=2, c='fuchsia', label=r"$||W_{B}^{tot}||$")
            else:
                plt.plot(losses / losses[0], linewidth=2, c='k', label="Loss")
                plt.plot(weights[:, -1]/weights[-1, -1], linewidth=2, c='b', label=r"$W_{A}^{tot}$")  # \frac{W_{A}^{tot}(t)}{W_{A}^{tot}(\infty)}
                plt.plot(weights[:, 0]/weights[-1, 0], linewidth=2, c='fuchsia', label=r"$W_{B}^{tot}$")  # \frac{W_{B}^{tot}(t)}{W_{B}^{tot}(\infty)}
                # plt.plot(np.arctan(weights[:, 0]/weights[:, 1]), color='b', alpha=0.7, label="angle")
                # plt.plot((weights[:, 0]**2+weights[:, 1]**2)/2, color='c', alpha=0.7, label="$|W_1|^2$")
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
                # plt.legend(loc='center right')
    plt.title(args.mode)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim((0, args.epoch))
    plt.yticks([0,0.5,1])

    # for superficial modality preference
    # plt.hlines(0.36, 0, args.epoch-20, 'grey', linestyles='dotted', linewidth=2)
    # plt.hlines(0.64, 0, args.epoch-20, 'k', linestyles='dotted', linewidth=2)
    # plt.yticks([0, 0.36, 0.64, 1], ['0', r'$\mathcal {L}(\mathcal {M}_B)$', r'$\mathcal {L}(\mathcal {M}_A)$', 1])
    # plt.xticks([0, 100, 200, 300])

    # for illustrating t_A, t_B
    # tA = time_half(args, weights[:, -1], True)
    # tB = time_half(args, weights[:, 0], True)
    # ylim = plt.gca().get_ylim()
    # plt.vlines(tA, ylim[0], np.max(weights[:, -1])/2, 'b', linestyles='dotted', linewidth=2)
    # plt.vlines(tB, ylim[0], 0.5, 'fuchsia', linestyles='dotted', linewidth=2)
    # plt.gca().set_ylim(ylim)
    # plt.xticks([0, tA, tB, args.epoch], ['0', '$t_A$', '$t_B$', args.epoch])
    # plt.gca().get_xticklabels()[1].set_color('b')
    # plt.gca().get_xticklabels()[2].set_color('fuchsia')
    # plt.hlines(0.75, 0, tA+30, 'b', linestyles='dotted', linewidth=2)
    # plt.hlines(1, 0, args.epoch, 'b', linestyles='dotted', linewidth=2)
    # plt.yticks([0, 1, 1.25], ['0', r'$W_{A}^{tot}(\infty)$', r'$W_{A}^{tot}(t_{uni})$'])

    plt.tight_layout(pad=0.5)

    if args.plot_Eg:
        plt.figure(figsize=(4, 3))
        plt.plot(results['Ls'], linewidth=2, c='k', label="Loss")
        plt.plot(results['Eg'], linewidth=2, c='r', label="$E_g$")
        plt.title('{} training samples'.format(args.dataset_size))
        plt.xlabel("Epoch")
        plt.xlim((0, args.epoch))
        plt.legend()
        plt.tight_layout(pad=0.5)
        # plt.savefig('img/early_Eg_{}train_noise.pdf'.format(args.dataset_size))
        # plt.savefig('img/Eg_{}train.jpg'.format(args.dataset_size), dpi=600)