import argparse

def config():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--loss", type=str, default='mse', choices={'mse', 'exponential', 'logistic'}, help='loss function')
    parser.add_argument("--epoch", type=int, default=1000, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.04, help='learning rate')
    parser.add_argument("--reg", type=float, default=0, help='regularization, i.e. weight_decay')
    parser.add_argument("--init", type=float, default=1e-12, help='weight initialization')
    parser.add_argument("--hid_width", type=int, default=1, help='number of hidden neurons')
    parser.add_argument("--relu", type=float, default=1, help='negative slope, 1 for linear, 0 for relu')
    parser.add_argument("--bias", action="store_true", help='bias or unbiased linear layer')

    # param for data set
    parser.add_argument("--data", type=str, default='toy', help='data type')
    parser.add_argument("--dataset_size", type=int, default=8192, help='number of training samples')
    parser.add_argument("--rho", type=float, default=0, help='Pearson correlation coefficient in toy dataset')
    parser.add_argument("--ratio", type=float, default=2, help='sigma_A / sigma_B ratio in toy dataset')
    parser.add_argument("--var_lin", type=float, default=1, help='variance of the linear modality in XOR dataset')
    parser.add_argument("--noise", type=float, default=0, help='std of noise in the output; 0 for noiseless')
    
    # param for network
    parser.add_argument("--mode", type=str, default='shallow', choices={'shallow', 'early_fusion', 'late_fusion', 'deep_fusion'}, help='model type')
    parser.add_argument("--depth", type=int, default=4, help='number of layers ')
    parser.add_argument("--fuse_depth", type=int, default=2, help='fuse at which layer')
    parser.add_argument("--sweep", type=str, default='single', choices={'single', 'depth_single', 'toy_sweep', 'rho_sweep', 'init_sweep', 'ratio_sweep', 'xor_sweep'}, help='sweep option')
    parser.add_argument("--repeat", type=int, default=1, help='number of repeats in sweep ')

    # param for logging settings
    parser.add_argument("--plot_weight", action="store_true", help="enable weights plot")
    parser.add_argument("--plot_Eg", action="store_true", help="enable generalization error plot")
    parser.add_argument("--vis_feat", action="store_true", help="enable feature visualization")
    parser.add_argument("--vis_contour", action="store_true", help="enable contour visualization")

    print(parser.parse_args(), '\n')
    return parser