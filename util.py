import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad


def integrand(u, wa, wb, L, Lf, u0):
    k = wb/wa
    if Lf == 2:
        ub = np.exp(k*np.log(u) + (1-k)*np.log(u0))
    else:
        ub = (k*u**(2-Lf) + (1-k)*u0**(2-Lf)) ** (1/(2-Lf))
    denominator = wa * u**(Lf-1) * (u**2 + ub**2) ** ((L-Lf)/2)
    return 1/denominator


def lag_depth(args):
    if args.fuse_depth == 1:
        return 1
    elif args.fuse_depth == args.depth:
        return 1 + (args.ratio**2 - 1) / (1 - args.rho**2)
    else:
        wa = args.ratio**2 + args.rho * args.ratio
        wb = 1 + args.rho * args.ratio
        L, Lf = args.depth, args.fuse_depth
        ua0 = args.init
        I = quad(integrand, ua0, 1, args=(wa, wb, L, Lf, ua0))
        if Lf == 2:
            ln_ub0 = np.log(1/args.init) * (1-wb/wa)
            lag = ln_ub0 * (1+args.rho/args.ratio)**(Lf/L-1) / (1-args.rho**2)
        else:
            ub0 = args.init * (1-wb/wa) ** (1/(2-Lf))
            lag = ub0**(2-Lf) * (1+args.rho/args.ratio)**(Lf/L-1) / ((Lf-2) * (1-args.rho**2))
        return 1 + lag/I[0]


def lag_twolayer(args, data):
    xa, xb, y = data['x1'], data['x2'], data['y']
    cov = data['cov']
    dim_a = xa.shape[1]
    y_xa = np.mean(y*xa, axis=0)
    y_xb = np.mean(y*xb, axis=0)
    cov_a = cov[0:dim_a, 0:dim_a]
    cov_ab = cov[0:dim_a, dim_a:]
    wa_uni = y_xa @ np.linalg.inv(cov_a)
    lag = (norm(y_xa) - norm(y_xb)) / norm(y_xb - wa_uni @ cov_ab)
    return 1 + lag, wa_uni


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X-mux
    Ymu = Y-muy
    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


def time_half(args, arr, weak=False):
    if weak or args.ratio == 1:
        half = np.max(arr) / 2
    else:
        half = (1 + args.rho / args.ratio) / 2
        if args.relu == 0:
            half *= 2
    res = (arr - half) ** 2
    return np.argmin(res)