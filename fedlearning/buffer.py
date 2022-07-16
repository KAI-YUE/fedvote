import copy
from collections import OrderedDict

# PyTorch Libraries
import torch
import torch.nn as nn

# My Libraries
from qnn import nn_registry
from qnn.networks import init_weights
from config.utils import *
from fedlearning.validate import *

class WeightBuffer(object):
    def __init__(self, weight_dict, mode="copy"):
        self._weight_dict = copy.deepcopy(weight_dict)
        if mode == "zeros":
            for w_name, w_value in self._weight_dict.items():
                self._weight_dict[w_name].data = torch.zeros_like(w_value)
        
    def __add__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data + weight_buffer._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def __sub__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data - weight_buffer._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def __mul__(self,rhs):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = rhs*self._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def push(self, weight_dict):
        self._weight_dict = copy.deepcopy(weight_dict)

    def state_dict(self):
        return self._weight_dict

