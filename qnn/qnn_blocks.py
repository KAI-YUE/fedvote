# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

# My Libraries
from config.loadconfig import load_config

class QuantizedConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(QuantizedConv2d, self).__init__(*kargs, **kwargs)
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        config = load_config()
        self.k = config.k

    def _init_latent_param(self):
        """Initialize the latent parameters
        """
        # initialize the latent variable
        self.weight.latent_param = torch.zeros_like(self.weight, requires_grad=True)

    def _apply(self, fn):
        # super(TernaryConv2d, self)._apply(fn)
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        pseudo_weight = torch.tanh(self.k*self.weight.latent_param)
        out = F.conv2d(input, pseudo_weight, self.bias,
                      self.stride, self.padding, self.dilation)
        return out

class QuantizedLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(QuantizedLinear, self).__init__(*kargs, **kwargs)
        self._init_latent_param()
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        config = load_config()
        self.k = config.k

    def _init_latent_param(self):
        """Initialize the placeholders for the multinomial distribution paramters.
        """
        # initialize the latent variable
        self.weight.latent_param = torch.zeros_like(self.weight, requires_grad=True)

    def _apply(self, fn):
        self.weight.latent_param.data = fn(self.weight.latent_param.data)
        self.bias.data = fn(self.bias.data)

        return self

    def forward(self, input):
        pseudo_weight = torch.tanh(self.k*self.weight.latent_param)
        out = F.linear(input, pseudo_weight, self.bias)

        return out
