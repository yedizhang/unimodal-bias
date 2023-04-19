import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from net import shallow_net, early_fusion, late_fusion, deep_fusion
from data import *
plt.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = '16'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.02, help='learning rate')
    parser.add_argument("--init", type=float, default=1e-12, help='weight initialization')
    parser.add_argument("--data", type=str, default='synergy', help='data type')
    parser.add_argument("--mode", type=str, default='shallow', help='model type')
    parser.add_argument("--hid_width", type=int, default=1, help='number of hidden neurons')
    parser.add_argument("--plot_weight", action="store_true", help="enable weights plot")
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    # param for deep_fusion mode
    parser.add_argument("--depth", type=int, default=6, help='number of layers ')
    parser.add_argument("--fuse_depth", type=int, default=2, help='fuse at which layer')
    return parser


def plot_training(args, losses, weights=None):
    from cycler import cycler
    plt.rc('axes', prop_cycle=(cycler('color', list(matplotlib.colors.BASE_COLORS))))
    # plt.figure(figsize=(10, 5))
    plt.plot(losses / losses[0], alpha=0.7, label=args.fuse_depth-1)
    # plt.plot(losses / losses[0], c='k', alpha=0.7)
    plt.xlabel("Epoch")
    plt.title(args.mode)
    if weights is not None:
        plt.plot(weights[:, 0], color='g', alpha=0.7, label="weight1")
        plt.plot(weights[:, 1], color='m', alpha=0.7, label="weight2")
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
    


def train(x1, x2, y, args):
    x = np.concatenate((x1, x2), -1)
    x1_tensor = torch.tensor(x1).float().to(device)
    x2_tensor = torch.tensor(x2).float().to(device)
    x_tensor = torch.tensor(x).float().to(device)
    y_tensor = torch.tensor(y[:,np.newaxis]).float().to(device)

    lr = args.lr  # Learning rate
    n_epochs = args.epoch  # Number of epochs
    dim_input = x_tensor.size(-1)
    dim_output = y_tensor.size(-1)

    if args.wandb:
        wandb.login()
        run = wandb.init(
            project="toy-multimodal",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": n_epochs,
            })

    # Model instantiation
    if args.mode == "shallow":
        network = shallow_net(dim_input, dim_output, args.init)
    elif args.mode == "early_fusion":
        network = early_fusion(dim_input, args.hid_width, dim_output, args.init)
    elif args.mode == "late_fusion":
        network = late_fusion(dim_input, args.hid_width, dim_output, args.init)
    elif args.mode == "deep_fusion":
        gamma = np.power(args.init, 1/(args.depth-1))
        print("gamma =", gamma)
        network = deep_fusion(dim_input, args.hid_width, dim_output, args.depth, args.fuse_depth, gamma)
    print(network)
    network = network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = np.zeros(n_epochs)  # loss records
    weights = np.zeros((n_epochs, dim_input)) if args.plot_weight else None
    # Training loop
    for i in range(n_epochs):  
        optimizer.zero_grad()
        if args.mode == "late_fusion" or args.mode == "deep_fusion":
            predictions= network(x1_tensor, x2_tensor)
        else:
            predictions = network(x_tensor)
        loss = criterion(predictions, y_tensor)
        # print(loss)
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()
        
        if args.plot_weight:
            model_weights = [param.data.detach().numpy() for param in network.parameters()]
            if args.mode == "shallow":
                weights[i, :] = model_weights[0]
            elif args.mode == "early_fusion":
                weights[i, :] = model_weights[1] @ model_weights[0]
            elif args.mode == "late_fusion":
                weights[i, 0] = model_weights[0] * model_weights[-1][0, 0]
                weights[i, 1] = model_weights[1] * model_weights[-1][0, 1]

            if args.wandb:
                wandb.log({"loss": loss.item(), "weight1": weights[i, 0], "weight2": weights[i, 1]})
    
    plot_training(args, losses, weights)


if __name__ == "__main__":
    parser = config()
    args = parser.parse_args()
    print(args, '\n')

    x1, x2, y = gen_toy_data()
    # vis_toy_data(x1, x2, y)

    # x1, x2, y = gen_data(args.data)
    
    plt.figure(figsize=(20, 5))
    for i in range(1, 7):
        args.fuse_depth = i
        train(x1, x2, y, args)
    plt.ylabel("Loss")
    plt.title("Loss records when fusing at different depth (total depth = {:d})".format(args.depth))
    plt.show()