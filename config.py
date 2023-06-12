import argparse

def config():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.02, help='learning rate')
    parser.add_argument("--reg", type=float, default=0, help='regularization, i.e. weight_decay')
    parser.add_argument("--init", type=float, default=1e-12, help='weight initialization')
    parser.add_argument("--hid_width", type=int, default=1, help='number of hidden neurons')
    parser.add_argument("--activation", type=str, default='linear', help='activation function')
    parser.add_argument("--bias", action="store_true", help='bias or unbiased linear layer')

    # data set
    parser.add_argument("--data", type=str, default='toy', help='data type')
    parser.add_argument("--dataset_size", type=int, default=4096, help='number of training samples')
    parser.add_argument("--mode", type=str, default='shallow', help='model type')
    parser.add_argument("--var_lin", type=float, default=1, help='variance of the linear modality in XOR dataset')

    # param for logging settings
    parser.add_argument("--plot_weight", action="store_true", help="enable weights plot")
    
    # param for deep_fusion mode
    parser.add_argument("--depth", type=int, default=6, help='number of layers ')
    parser.add_argument("--fuse_depth", type=int, default=2, help='fuse at which layer')

    print(parser.parse_args(), '\n')
    return parser