import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rc('font', family="Arial")
plt.rcParams['font.size'] = '14'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from net import shallow_net, early_fusion, late_fusion, deep_fusion, fission
from config import config
from util import *
from numerical import *
from data import *
from vis import *
from sweep import *


def creat_network(args, in_dim, out_dim):
    # Model instantiation
    if args.mode == "shallow":
        network = shallow_net(in_dim, out_dim, args.init)
    elif args.mode == "early_fusion":
        network = early_fusion(in_dim, args.hid_width, out_dim, args.activation, args.bias, args.init)
    elif args.mode == "late_fusion":
        network = late_fusion(in_dim, args.hid_width, out_dim, args.activation, args.bias, args.init)
    elif args.mode == "deep_fusion":
        # gamma = np.power(args.init, 1/(1+args.fuse_depth))
        # print("gamma =", gamma)   # init should scale with init1/init2 = exp(depth2/depth1)
        network = deep_fusion(in_dim, args.hid_width, out_dim, args.depth, args.fuse_depth, args.activation, args.init)
    elif args.mode == "fission":
        network = fission(in_dim, args.hid_width, out_dim, args.bias, args.init)
    print(network)
    return network.to(device)


def unpack_weights(parameters, args, w_dim, in_dim):
    hid = args.hid_width
    Lf = args.fuse_depth - 1
    W_tot = np.ones(w_dim)
    W = [param.data.cpu().detach().numpy() for param in parameters]
    if args.mode == "shallow":
        W_tot = W[0].squeeze()
        in_hid = W[0]
    elif args.mode == "early_fusion":
        W_tot = (W[1] @ W[0]).squeeze()
        in_hid = W[0]
        # W1 = W[0]
        # W2 = W[1]
        # W_gt = np.array([1,2])
        # W_gt = W_gt / np.linalg.norm(W_gt)
        # M = W1.T @ W1 + np.linalg.norm(W2) * np.eye(2)
        # align = W_gt.T @ M @ W_gt / np.linalg.norm(M)
        # print(np.linalg.eigvals(W1.T @ W1).real)
        # print(np.linalg.eigvals(M).real)
        # print(align)
    elif args.mode == "late_fusion":
        W_tot[:in_dim[0]] = W[-1][:, :hid] @ W[0]
        W_tot[in_dim[0]:] = W[-1][:, hid:] @ W[1]
        in_hid = np.concatenate((W[0], W[1]), -1)
    elif args.mode == "deep_fusion":
        if args.fuse_depth == 1:  # deep early fusion
            in_hid = W[0]
            W_tot = np.eye(w_dim)
            for i in range(len(W)):
                W_tot = W[i] @ W_tot
        else:  # deep late fusion
            in_hid = np.concatenate((W[0], W[1]), -1)
            h1, h2 = np.eye(in_dim[0]), np.eye(in_dim[1])
            for i in range(0, Lf):
                h1, h2 = W[2*i] @ h1, W[2*i+1] @ h2
                # print(np.linalg.norm(W[2*i])**2 + np.linalg.norm(W[2*i+1])**2)
            h = np.concatenate((W[2*Lf][:, :hid] @ h1, W[2*Lf][:, hid:] @ h2), -1)
            # d = np.eye(h.shape[0])
            for i in range(2*Lf+1, len(W)):
                # print(np.linalg.norm(W[i])**2)
                h = W[i] @ h
                # d = W[i] @ d
            # print(np.linalg.norm(d))
            W_tot = h
            # W_tot[:,0] = np.linalg.norm(W[1])
            # W_tot[:,1] = np.linalg.norm(W[0])
    elif args.mode == "fission":
        W_tot[0] = W[1] @ W[0][:hid, :] 
        W_tot[1] = W[2] @ W[0][hid:, :] 
        in_hid = W[0]
    return W_tot, in_hid


def train(data, args):
    if args.mode == "late_fusion" or args.mode == "deep_fusion":
        x1_tensor, x2_tensor, y_tensor, in_dim, out_dim = prep_data(args, data, device)
    else:
        x_tensor, y_tensor, in_dim, out_dim = prep_data(args, data, device)
    network = creat_network(args, in_dim, out_dim)
    optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.reg)

    losses = np.zeros(args.epoch)  # loss records
    w_dim = in_dim if isinstance(in_dim, int) else sum(in_dim)
    weights = np.zeros((args.epoch, w_dim)) if args.plot_weight else None
    if args.vis_feat:
        fig, axs, ims = prep_axs(args)
    # Training loop
    for i in range(args.epoch):
        optimizer.zero_grad()
        if args.mode == "late_fusion" or args.mode == "deep_fusion":
            predictions= network(x1_tensor, x2_tensor)
        else:
            predictions = network(x_tensor)
        loss = 0.5*nn.MSELoss()(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()
        
        if args.plot_weight:
            weights[i, :], feat = unpack_weights(network.parameters(), args, w_dim, in_dim)
            if i % 1000 == 0:
                print(i, losses[i])
                fig_feat(args, feat, i)
            if args.vis_feat and i % 10 == 0:
                data_res = data.copy()
                data_res['y'] = data['y'] - predictions.cpu().detach().numpy()
                if args.vis_contour:
                    axs[1].cla()
                    axs[2].cla()
                    ims.append(vis_relu(args, data_res, feat, losses[:i], axs))
                    # plt.savefig('frame/{:04d}.jpg'.format(i), dpi=300)
                else:
                    ims.append(vis_relu(args, data_res, feat, losses[:i], axs))
        if args.sweep != 'single' and losses[i] < 10e-5:
            losses, weights = losses[:i], weights[:i]
            print("Converged at epoch ", i)
            break

    # count = count_angle(cart2sph(feat))
    # vis_relu_3d(feat)

    if args.vis_feat:
        ani = animation.ArtistAnimation(fig, ims, interval=20, blit=False)
        fig.colorbar(ims[1][0], orientation='vertical')
        plt.tight_layout()
        ani.save('early_relu_+-1xor_100hid.mp4', dpi=300)
        plt.show()
    else:
        if args.sweep == 'single' or 'depth_single':
            plot_training(args, losses, weights)
        return losses, weights


if __name__ == "__main__":
    args = config().parse_args()
    if args.sweep == 'single':
        data = gen_data(args)
        train(data, args)
    elif args.sweep == 'depth_single':
        depth_single(args)
    elif args.sweep == 'toy_sweep':
        toy_sweep(args)
    elif args.sweep == 'depth_sweep':
        depth_sweep(args)
    elif args.sweep == 'ratio_sweep':
        ratio_sweep(args)
    elif args.sweep == 'init_sweep':
        init_sweep(args)
    elif args.sweep == 'xor_sweep':
        xor_sweep(args)