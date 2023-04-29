import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from net import shallow_net, early_fusion, late_fusion, deep_fusion
from data import *
from config import config
plt.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = '16'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_training(args, losses, weights=None):
    plt.figure(figsize=(10, 5))
    plt.plot(losses / losses[0], c='k', alpha=0.7, label="loss")
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
    plt.show()
    

def train(x1, x2, y, args):
    x = np.concatenate((x1, x2), -1)
    x1_tensor = torch.tensor(x1).float().to(device)
    x2_tensor = torch.tensor(x2).float().to(device)
    x_tensor = torch.tensor(x).float().to(device)
    y_tensor = torch.tensor(y[:,np.newaxis]).float().to(device)

    dim_input = x_tensor.size(-1)
    dim_output = y_tensor.size(-1)

    if args.wandb:
        wandb.login()
        run = wandb.init(
            project="toy-multimodal",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "regularization": args.reg,
                "epochs": args.epoch,
            })

    # Model instantiation
    if args.mode == "shallow":
        network = shallow_net(dim_input, dim_output, args.init)
    elif args.mode == "early_fusion":
        network = early_fusion(dim_input, args.hid_width, dim_output, args.activation, args.bias, args.init)
    elif args.mode == "late_fusion":
        network = late_fusion(dim_input, args.hid_width, dim_output, args.activation, args.bias, args.init)
    elif args.mode == "deep_fusion":
        gamma = np.power(args.init, 1/(args.depth-1))
        print("gamma =", gamma)
        network = deep_fusion(dim_input, args.hid_width, dim_output, args.depth, args.fuse_depth, gamma)
    print(network)
    network = network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.reg)

    # optimizer = optim.SGD([
    #                         {'params': network.hid_out.parameters()},
    #                         {'params': network.inB_hid.parameters()},
    #                         {'params': network.inA_hid.parameters(), 'lr': args.lr*16}
    #                       ], lr=args.lr)

    criterion = nn.MSELoss()

    losses = np.zeros(args.epoch)  # loss records
    weights = np.zeros((args.epoch, dim_input)) if args.plot_weight else None
    # Training loop
    for i in range(args.epoch):  
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
            model_weights = [param.data.cpu().detach().numpy() for param in network.parameters()]
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
    args = config().parse_args()
    print(args, '\n')

    if args.data == 'toy':
        x1, x2, y = gen_toy_data(noise=False, size=args.dataset_size)
        # vis_toy_data(x1, x2, y)
    elif args.data == 'xor':
        x1, x2, y = gen_xor_data()
    else:
        x1, x2, y = gen_data(args.data)

    train(x1, x2, y, args)