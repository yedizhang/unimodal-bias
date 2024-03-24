from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import CNN, FCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        if batch_idx == 0:
            Ls = loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        loss.backward()
        optimizer.step()
    return Ls


def test(model, device, test_loader):
    Eg = test_unit(model, device, test_loader)
    Eg_A = test_unit(model, device, test_loader, 'A')
    Eg_B = test_unit(model, device, test_loader, 'B')
    return Eg, Eg_A, Eg_B


def test_unit(model, device, test_loader, unimodal=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, unimodal)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\nTest set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        unimodal, test_loss, correct, len(test_loader.dataset), 100. * test_acc))
    return test_acc


def vis(args, Ls, Eg, Eg_A, Eg_B):
    filename = "mnist_L{}_Lf{}_seed{}".format(args.depth, args.fuse_depth, args.seed)
    import pandas as pd
    df = pd.DataFrame({'Ls': Ls,
                       'Eg': Eg,
                       'Eg_A': Eg_A,
                       'Eg_B': Eg_B})
    df.to_csv("{}.csv".format(filename))

    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.figure(figsize=(4, 3))
    plt.plot(Ls/Ls[0], c='k', linewidth=2, label="Loss")
    plt.plot(Eg_A, c='b', linewidth=2, label="A acc")
    plt.plot(Eg_B, c='fuchsia', linewidth=2, label="B acc")
    plt.plot(Eg, 'k--', linewidth=2, label="A&B acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.xlim((0, args.epoch-1))
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("{}.svg".format(filename))


def config():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epoch', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.04, metavar='LR',
                        help='learning rate (default: 0.04)')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='M',
                        help='Learning rate step gamma (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument("--depth", type=int, default=5, help='number of layers ')
    parser.add_argument("--fuse_depth", type=int, default=2, help='fuse at which layer')
    print(parser.parse_args(), '\n')
    return parser


def mnist():
    args = config().parse_args()
    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    Ls = np.zeros(args.epoch)
    Eg, Eg_A, Eg_B = np.copy(Ls), np.copy(Ls), np.copy(Ls)

    model = FCN(depth=args.depth, fuse_depth=args.fuse_depth).to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epoch + 1):
        Eg[epoch-1], Eg_A[epoch-1], Eg_B[epoch-1] = test(model, device, test_loader)
        Ls[epoch-1] = train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist.pt")

    vis(args, Ls, Eg, Eg_A, Eg_B)


if __name__ == '__main__':
    mnist()