import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = '16'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from net import shallow_net, early_fusion, late_fusion, deep_fusion, fission
from config import config
from util import *
from numerical import *
from data import *
from vis import *


def train(data, args):
    if args.mode == "late_fusion" or args.mode == "deep_fusion":
        x1_tensor = torch.tensor(data['x1']).float().to(device)
        x2_tensor = torch.tensor(data['x2']).float().to(device)
        y_tensor = torch.tensor(data['y']).float().to(device)
        in_dim = [x1_tensor.size(-1), x2_tensor.size(-1)]
        out_dim = y_tensor.size(-1)
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
    network = network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.reg)

    losses = np.zeros(args.epoch)  # loss records
    w_dim = in_dim if isinstance(in_dim, int) else sum(in_dim)
    weights = np.zeros((args.epoch, w_dim)) if args.plot_weight else None
    if args.plot_weight and args.data == 'xor':
        ims = []
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(args.mode+' ReLU net, $\pm$1XOR & Gaussian var='+str(args.var_lin))
        ax1 = fig.add_subplot(1,3,1)
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlim([0, args.epoch])
        ax2 = fig.add_subplot(1,3,2)
        ax2.grid()
        ax2.axis('equal')
        ax3 = fig.add_subplot(1,3,3, projection='3d')
        plt.tight_layout()
    # Training loop
    for i in range(args.epoch):
        optimizer.zero_grad()
        if args.mode == "late_fusion" or args.mode == "deep_fusion":
            predictions= network(x1_tensor, x2_tensor)
        else:
            predictions = network(x_tensor)
        loss = nn.MSELoss()(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()

        # if args.data == 'xor' and i == args.epoch-1:
        #     model_weights = [param.data.cpu().detach().numpy() for param in network.parameters()]
        #     if args.mode == "early_fusion":
        #         W = model_weights[0]
        #     elif args.mode == "late_fusion":
        #         W = np.concatenate((model_weights[0], model_weights[1]), -1)
        #     count = count_angle(cart2sph(W))
        #     # vis_relu_3d(W)
        
        if args.plot_weight:
            model_weights = [param.data.cpu().detach().numpy() for param in network.parameters()]
            if args.data != 'xor':
                if args.mode == "shallow":
                    weights[i, :] = model_weights[0].squeeze()
                elif args.mode == "early_fusion":
                    weights[i, :] = (model_weights[1] @ model_weights[0]).squeeze()
                elif args.mode == "late_fusion":
                    weights[i, 0] = model_weights[0] * model_weights[-1][0, 0]
                    weights[i, 1] = model_weights[1] * model_weights[-1][0, 1]
                elif args.mode == "fission":
                    weights[i, 0] = model_weights[0][0] * model_weights[1]
                    weights[i, 1] = model_weights[0][1] * model_weights[2]
            elif args.data == 'xor' and i % 20 == 0:
                if args.mode == "early_fusion":
                    W = model_weights[0]
                elif args.mode == "late_fusion":
                    W = np.concatenate((model_weights[0], model_weights[1]), -1)
                data_res = data.copy()
                data_res['y'] = data['y'] - predictions.cpu().detach().numpy().squeeze()
                ax2.cla()
                ax3.cla()
                ims.append(vis_relu(data_res, W, losses[:i], ax1, ax2, ax3))
                plt.savefig('frame/{:04d}.jpg'.format(i), dpi=300)
    
    if args.plot_weight and args.data == 'xor':
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)
        fig.colorbar(ims[1][0], orientation='vertical')
        plt.tight_layout()
        ani.save('early_relu_+-1xor_100hid.mp4', dpi=300)
        plt.show()
    else:
        plot_training(args, losses, weights)


if __name__ == "__main__":
    args = config().parse_args()
    
    data = gen_data(args)
    
    if args.mode == "deep_fusion":
        plt.figure(figsize=(10, 5))
        for i in range(1, args.depth+1):
            args.fuse_depth = i
            train(data, args)
        plt.title("Loss records when fusing at different depth (total depth = {:d})".format(args.depth))
        plt.ylabel("Loss")
    else:
        train(data, args)

    plt.show()