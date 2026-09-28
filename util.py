import math
import numpy as np
from pathlib import Path
import random
import torch
import torch.nn as nn
import sys


def set_random_seed(seed):
    """Seed Python, NumPy, and PyTorch without forcing slower deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_paths(**paths):
    """Raise one concise error listing required input paths that do not exist."""
    missing = [f'{name}={path}' for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('Required input path(s) not found: ' + ', '.join(missing))


class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


# -*- coding: utf-8 -*-

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"w")

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


_DEFAULT_ALPHA = 1e-6

class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return _sigmoid(x, logpx, self.alpha)
        else:
            return _logit(x, logpx, self.alpha)


class SigmoidTransform(nn.Module):
    """Reverse of LogitTransform."""

    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return _logit(x, logpx, self.alpha)
        else:
            return _sigmoid(x, logpx, self.alpha)


def _logit(x, logpx=None, alpha=_DEFAULT_ALPHA):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if logpx is None:
        return y
    return y, logpx - _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _sigmoid(y, logpy=None, alpha=_DEFAULT_ALPHA):
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    if logpy is None:
        return x
    return x, logpy + _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad


def calc_z_shapes(n_channel, input_size, n_levels):
    z_shapes = []
    for i in range(n_levels - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes


def gen_noise(num_img, sigma, img_size, n_levels, device):
    z_sample = []
    z_shapes = calc_z_shapes(3, img_size, n_levels)
    for z in z_shapes:
        z_new = torch.randn(num_img, *z) * sigma
        z_sample.append(z_new.to(device))
    return z_sample


def process_img(x, n_bits):    # De-quantize and convert to logits
    n_bins = 2**n_bits
    x = torch.floor(x * 255 / 2 ** (8 - n_bits))
    x = x / n_bins
    x = x +  torch.rand_like(x) / n_bins
    return x

