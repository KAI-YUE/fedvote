from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.nn as nn

class QuantizedNN(ABC, nn.Module):
    def __init__(self):
        super(QuantizedNN, self).__init__()

    def latent_parameters(self):
        for module in self.modules():
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            yield module.weight.latent_param

    def latent_param_dict(self):
        lparam_dict = OrderedDict()
        named_modules = self.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue

            lparam_dict[module_name + ".weight.latent_param"] = module.weight.latent_param

        return lparam_dict

    def load_latent_param_dict(self, lparam_dict):
        for latent_param_name, latent_param in lparam_dict.items():
            exec("self.{:s} = latent_param".format(latent_param_name))

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

from qnn.qnn_blocks import *
from qnn.qnn_struct import *
from qnn.networks import *
from qnn.dataset import UserDataset
from qnn.initialize import *
