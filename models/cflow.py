from atexit import register
import math

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

logabs = lambda x: torch.log(torch.abs(x))

class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x, y=None):
        B, C, H, W = x.shape
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.bias)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1) + 1e-6).view_as(self.scale)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - logabs(self.scale).sum() * H * W
        return z, logdet

    def inverse(self, z, y=None):
        return z * self.scale + self.bias, logabs(self.scale).sum() * z.shape[2] * z.shape[3]


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer"""
    def __init__(self, n_channels=3, lu_factorize=True):
        super().__init__()
        self.lu_factorize = lu_factorize

        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.linalg.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            A_LU, pivots = w.lu()
            wp, wl, wu = torch.lu_unpack(A_LU, pivots)
            
            s = torch.diag(wu)  # calculate diagnal values
            log_s = logabs(s)  
            sign_s = torch.sign(s) 

            l_mask = torch.tril(torch.ones_like(wl), -1)
            wu = torch.triu(wu, 1)  # Compute the upper triangular matrix

            self.register_buffer("p", wp)
            self.register_buffer("l_mask", l_mask)
            self.register_buffer("sign_s", sign_s)
            self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))

            self.l = nn.Parameter(wl)
            self.u = nn.Parameter(wu)
            self.log_s = nn.Parameter(log_s)

        else:
            self.w = nn.Parameter(w)

    def cal_weight(self):
        l = self.l * self.l_mask + self.l_eye
        u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
        weight = self.p @ l @ u
        return weight

    def forward(self, x, y=None):
        B,C,H,W = x.shape
        if self.lu_factorize:
            weight = self.cal_weight()
            logdet = self.log_s.sum() * H * W
        else:
            weight = self.w
            logdet = torch.slogdet(weight)[-1] * H * W
        return F.conv2d(x, weight.view(C,C,1,1)), logdet

    def inverse(self, z, y=None):
        B,C,H,W = z.shape
        if self.lu_factorize:
            weight = self.cal_weight()
            w_inv = torch.inverse(weight)
            logdet = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(w_inv)[-1] * H * W
        return F.conv2d(z, w_inv.view(C,C,1,1)), logdet


class ConditionAffine(nn.Module):   # adding conditions
    """ Conditional Affine coupling layer"""
    def __init__(self, n_channels, width, n_y):
        super().__init__()

        # network layers;
        self.conv1 = nn.Conv2d(n_channels//2, width, kernel_size=3, padding=1, bias=False) 
        self.norm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.cond1 = nn.Linear(n_y, width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.norm2 = Actnorm(param_dim=(1, width, 1, 1))

        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3, padding=1)    # output is split into scale and shift components
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels,1,1))   # learned scale (cf RealNVP sec 4.1 / Glow official code

    def forward(self, x, y):
        x_a, x_b = x.chunk(2, 1)  # split along channel dim

        h0 = self.conv1(x_b) + self.cond1(y.float()).unsqueeze(-1).unsqueeze(-1)
        h0 = F.relu(self.norm1(h0)[0])

        h0 = self.conv2(h0)
        h0 = F.relu(self.norm2(h0)[0])      # shape (B, width, H, W)       
        #-------------------------------------------------------------------------
        h = self.conv3(h0) * self.log_scale_factor.exp()

        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        logdet = s.log().sum([1, 2, 3])

        return z, logdet

    def inverse(self, z, y):
        z_a, z_b = z.chunk(2, 1)  # split along channel dim

        h0 = self.conv1(z_b) + self.cond1(y.float()).unsqueeze(-1).unsqueeze(-1)
        h0 = F.relu(self.norm1(h0)[0])

        h0 = self.conv2(h0)
        h0 = F.relu(self.norm2(h0)[0])      # shape (B, width, H, W)       
        #-------------------------------------------------------------------------
        h = self.conv3(h0) * self.log_scale_factor.exp()

        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        logdet = - s.log().sum([1, 2, 3])

        return x, logdet

class Squeeze(nn.Module):  # 
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of 
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)   # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)         # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4*C, H//2, W//2)       # aggregate spatial dim factors into channels
        return x

    def inverse(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C//4, 2, 2, H, W)      # factor channel dim
        x = x.permute(0, 1, 4, 2, 5, 3)         # transpose to (B, C//4, H, 2, W, 2)
        x = x.reshape(B, C//4, 2*H, 2*W)        # aggregate channel dim factors into spatial dims
        return x


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """
    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels//2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return x2, logdet


class Prior(nn.Module):
    """
    Conditonal image synthesis on  given labels
    """
    def __init__(self, n_y, n_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)
        self.loc = nn.Linear(n_y, 2*n_channels)
        
        self.conv.weight.data.zero_()
        self.loc.weight.data.zero_()

    def forward(self, z, y):
        h = self.conv(torch.zeros_like(z))
        y_cond = self.loc(y.float())
        h += y_cond.view(h.shape[0], h.shape[1], 1, 1)

        mu, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z = (z - mu) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z, logdet

    def inverse(self, z, y):
        h = self.conv(torch.zeros_like(z))
        y_cond = self.loc(y.float())
        h += y_cond.view(h.shape[0], h.shape[1], 1, 1)

        mu, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        z = mu + z * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return z, logdet

class Preprocess(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bin = 2**bits

    def forward(self, x):
        logdet = - math.log(self.bin) * x[0].numel() 
        return x - 0.5, logdet                 

    def inverse(self, x):
        logdet = math.log(self.bin) * x[0].numel()
        return x + 0.5, logdet

# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, y):
        sum_logdets = 0.
        for module in self:
            x, logdet = module(x, y) 
            sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    def inverse(self, z, y):
        sum_logdets = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z, y)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels, width, n_y, lu_factorize=True):
        super().__init__(Actnorm(param_dim=(1,n_channels,1,1)),
                         Invertible1x1Conv(n_channels, lu_factorize),
                         ConditionAffine(n_channels, width, n_y))


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """
    def __init__(self, n_channels, width, depth, n_y, lu_factorize=False):
        super().__init__()
        # network layers
        self.Squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4*n_channels, width, n_y, lu_factorize) for _ in range(depth)])
        self.split = Split(4*n_channels)

    def forward(self, x, y):
        x = self.Squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x, y)
        x1, z2, logdet_split = self.split(x)
        logdet = logdet_flowsteps + logdet_split
        return x1, z2, logdet

    def inverse(self, x1, z2, y):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x, y)
        x = self.Squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet


# --------------------
# Model
# --------------------

class CFlow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""
    def __init__(self, bits, width, depth, n_levels, input_dims=(3,32,32), n_y=10, lu_factorize=True):
        super().__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * 4**n_levels / 2**(n_levels-1))  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / 2**n_levels)                                # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)
        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess(bits)

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x (n_levels-1) -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * 2**i, width, depth, n_y, lu_factorize) for i in range(n_levels-1)])
        self.Squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels, width, n_y, lu_factorize) for _ in range(depth)])

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # gaussianize the final z output based on conditions
        self.prior = Prior(n_y, out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss()

        self.project_y = nn.Sequential(nn.Linear(out_channels, out_channels), nn.Linear(out_channels, n_y))

    def forward(self, x, n_y=10, y=None):
        x, sum_logdets = self.preprocess(x)
        # y_onehot = F.one_hot(y, num_classes=n_y)
        # pass through flow
        zs = []
        for m in self.flowlevels:
            x, z, logdet = m(x, y)
            sum_logdets = sum_logdets + logdet
            zs.append(z)
        x = self.Squeeze(x)
        z, logdet = self.flowstep(x, y)
        sum_logdets = sum_logdets + logdet

        
        # gaussianize the final z
        if y is None:
            z, logdet = self.gaussianize(torch.zeros_like(z), z)
        else:
            z, logdet = self.prior(z, y)

        sum_logdets = sum_logdets + logdet
        zs.append(z)

        # # the conditional loss 
        # y_logit = self.project_y(torch.mean(z, dim=[2,3]))

        # return zs, sum_logdets, y_logit
        return zs, sum_logdets

    def inverse(self, zs=None, n_y=10, y=None, batch_size=None, z_std=1.):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        # pass through inverse flow
        if y is None:
            z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])
        else:
            # y_onehot = F.one_hot(y, num_classes=n_y)
            z, sum_logdets = self.prior.inverse(zs[-1], y)
        x, logdet = self.flowstep.inverse(z, y)
        sum_logdets = sum_logdets + logdet
        x = self.Squeeze.inverse(x)
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            x, logdet = m.inverse(x, z, y)
            sum_logdets = sum_logdets + logdet
        # postprocess
        x, logdet = self.preprocess.inverse(x)
        sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def cal_loss(self, x, n_y, y, loss_weight, bits_per_pixel=False):
        zs, logdet, y_logit = self.forward(x, n_y, y)
        log_prob = sum(self.base_dist.log_prob(z).sum([1,2,3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())

        # y_loss = self.CE(y_logit, y)
        # total_loss = - log_prob + y_weight * y_loss
        # y_predict = torch.argmax(y_logit, dim=-1)
        # class_error = 1 -  torch.eq(y_predict, y).sum() / np.prod(y.shape)
        y_loss = self.BCE(y_logit, y.float())
        total_loss = - log_prob + loss_weight * y_loss.mean(-1)
        y_predict = torch.sigmoid(y_logit)
        class_error = 1 - torch.eq(y_predict > 0.5, y).sum() / np.prod(y.shape)

        return total_loss, class_error
