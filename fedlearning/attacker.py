import numpy as np
import copy
from collections import OrderedDict

# Pytorch Libraries
import torch

# My Libraries 
from qnn.networks import init_weights

def generate_attacker_list(config):
    user_ids = np.arange(config.users)
    np.random.shuffle(user_ids)
    attacker_list = user_ids[:config.num_attackers]
    normal_user_ids = user_ids[config.num_attackers:]

    return dict(attacker_list=attacker_list, 
                normal_user_ids=normal_user_ids)

def random_bnn_package(model):
    laten_param = model.latent_param_dict()
    random_sampled_weight = OrderedDict()
    latent_param_name_len = 13
    for param_name, param_value in laten_param.items():
        ones_tensor = torch.ones_like(param_value)
        rand_variable = torch.randint_like(param_value, high=2)
        rand_variable = torch.where(rand_variable==0, -ones_tensor, rand_variable)
        
        weight_name = param_name[:-latent_param_name_len]
        random_sampled_weight[weight_name] = rand_variable
    
    return random_sampled_weight