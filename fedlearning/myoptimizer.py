import numpy as np
import logging
import copy
from collections import OrderedDict

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
from fedlearning.buffer import WeightBuffer
from fedlearning.validate import *
from qnn import UserDataset
from config.utils import *

class LocalUpdater(object):
    def __init__(self, user_resource, config, **kwargs):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   a dictionary containing images and labels listed as follows. 
                - images (ndarry):  training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - mode (int):       the mode indicating the local model type.
                - device (str):     set 'cuda' or 'cpu' for the user. 
        """
        
        try:
            self.lr = user_resource["lr"]
            self.momentum = user_resource["momentum"]
            self.weight_decay = user_resource["weight_decay"]
            self.batch_size = user_resource["batch_size"]
            self.device = user_resource["device"]
            
            assert("images" in user_resource)
            assert("labels" in user_resource)
        except KeyError:
            logging.error("LocalUpdater initialization failure! Input should include `lr`, `batch_size`!") 
        except AssertionError:
            logging.error("LocalUpdater initialization failure! Input should include samples!") 

        self.mode = config.mode
        self.local_weight = None
        dataset_type = parse_dataset_type(config)

        if config.imbalance:
            sampler = WeightedRandomSampler(user_resource["sampling_weight"], 
                                    num_samples=user_resource["sampling_weight"].shape[0])

            self.sample_loader = \
                DataLoader(UserDataset(user_resource["images"], 
                                user_resource["labels"],
                                dataset_type), 
                            sampler=sampler,
                            # sampler=None,
                            batch_size=self.batch_size)
        else:
            self.sample_loader = \
                DataLoader(UserDataset(user_resource["images"], 
                            user_resource["labels"],
                            dataset_type), 
                            sampler=None, 
                            shuffle=True,
                            batch_size=self.batch_size)

        self.criterion = nn.CrossEntropyLoss()

        self.tau = config.tau
        self.k = config.k

    def local_step(self, model,  **kwargs):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model.
        """
        # if we are training a full precision network
        # the copy of the model is state dict
        if self.mode == 0 or self.mode == 1:
            w_copy = copy.deepcopy(model.latent_param_dict())
            # optimizer = optim.SGD(model.latent_parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            optimizer = optim.Adam(model.latent_parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # optimizer = optim.AdamW(model.latent_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.mode == 2:
            w_copy = copy.deepcopy(model.state_dict())
            # optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        tau_counter = 0
        break_flag = False

        while not break_flag:
            for sample in self.sample_loader:
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)
                optimizer.zero_grad()

                output = model(image)
                loss = self.criterion(output, label)

                loss.backward()
                optimizer.step()

                tau_counter += 1
                if tau_counter >= self.tau:
                    break_flag = True
                    break
                
        # test_acc = test_bnn_accuracy(model, kwargs["test_data"], kwargs["config"])
        # print("local acc {:.3f}".format(test_acc))

        if self.mode == 0 or self.mode == 1:
            self.local_weight = copy.deepcopy(model.latent_param_dict())
            model.load_latent_param_dict(w_copy)
        else:
            self.local_weight = copy.deepcopy(model.state_dict())
            model.load_state_dict(w_copy)

    def uplink_transmit(self):
        """Simulate the transmission of local weights to the central server.
        """ 
        try:
            assert(self.local_weight != None)
        except AssertionError:
            logging.error("No local model is in the buffer!")

        # sample a ternary weight
        if self.mode == 0:
            latent_param_name_len = 13
            sampled_qnn_state_dict = OrderedDict()
            for param_name, param_value in self.local_weight.items():
      
                # probability vector [pr(w_b=-1), pr(w_b=0), pr(w_b=1)] 
                pseudo_weight = torch.tanh(self.k*param_value)
                idx_m1 = (pseudo_weight < 0)
                idx_p1 = (pseudo_weight > 0)
                
                weight_shape = torch.tensor(pseudo_weight.shape).tolist()
                weight_shape.append(3)
                prob = torch.zeros(weight_shape)
                prob = prob.to(pseudo_weight.device)

                prob[..., 0][idx_m1] = pseudo_weight[idx_m1].abs()
                prob[..., 2][idx_p1] = pseudo_weight[idx_p1].abs()
                prob[..., 1] = 1 - prob[..., 0] - prob[..., 1]
                
                # random sampling
                ones_tensor = torch.ones_like(pseudo_weight)
                random_variable = torch.rand_like(pseudo_weight)
                sampled_weight = torch.zeros_like(pseudo_weight)
                sampled_weight = torch.where(random_variable < prob[..., 0], -ones_tensor, sampled_weight)
                sampled_weight = torch.where(1 - prob[...,2] < random_variable, ones_tensor, sampled_weight)
                
                # map sampling
                # sampled_weight = torch.argmax(prob, dim=-1)
                # sampled_weight -= 1

                weight_name = param_name[:-latent_param_name_len]
                sampled_qnn_state_dict[weight_name] = sampled_weight.clone()
                
            local_package = sampled_qnn_state_dict
        
        # sample the binary weights
        elif self.mode == 1:
            latent_param_name_len = 13
            sampled_bnn_state_dict = OrderedDict()
            for param_name, param_value in self.local_weight.items():
                ones_tensor = torch.ones_like(param_value.data)
                
                pseudo_weight = torch.tanh(self.k*param_value.data)
                prob_eq_p1 = (pseudo_weight + 1)/2
                random_variable = torch.rand_like(prob_eq_p1)
                sampled_weight = torch.where(random_variable < prob_eq_p1, ones_tensor, -ones_tensor)

                # max likelihood sampling
                # sampled_weight = param_value.sign()

                weight_name = param_name[:-latent_param_name_len]
                sampled_bnn_state_dict[weight_name] = sampled_weight.clone()

            local_package = sampled_bnn_state_dict

        elif self.mode == 2:
            local_package = self.local_weight

        return local_package

class GlobalUpdater(object):
    def __init__(self, config, initial_model, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
                - predictor (str):       predictor type.
                - quantizer (str):       quantizer type.
                - entropy_encoder (str): entropy encoder type.

            initial_model (OrderedDict): initial model state_dict
        """
        self.num_users = int(config.users * config.sampling_fraction)
        self.mode = config.mode 
        self.device = config.device

        if self.mode == 0 or self.mode == 1:
            self.global_latent_param = copy.deepcopy(initial_model.latent_param_dict())
        elif self.mode == 2:
            self.global_weight = copy.deepcopy(initial_model.state_dict())    

        self.k = config.k
        
        self.beta = 0.5
        self.user_previous_credibility = np.ones(self.num_users)
        self.user_credibility = np.ones(self.num_users)
        self.vote_weight = np.ones(self.num_users)
        self.total_votes = None

    def global_step(self, model, local_packages, **kwargs):
        idx = list(local_packages.keys())[0]
        accumulated_weight = WeightBuffer(local_packages[idx], "zeros")
        for user_id, package in local_packages.items():
            accumulated_weight = WeightBuffer(package) + accumulated_weight

        if self.mode == 0 or self.mode == 1:
            accumulated_weight_state_dict = accumulated_weight.state_dict()
            voted_results = OrderedDict()

            if self.mode == 0:
                # set counters for -1, 0, 1; 
                # toy example: {0, 0, 0, 1} and {-1, 1, 0, 1}, the aggregated results cannot tell the difference  
                num_ones = OrderedDict()
                num_zeros = OrderedDict()
                for w_name, w_value in local_packages[idx].items():
                    num_ones[w_name] = torch.zeros_like(w_value)
                    num_zeros[w_name] = torch.zeros_like(w_value)
                    voted_results[w_name] = -torch.ones_like(w_value)

                for user_id, package in local_packages.items():
                    for w_name, w_value in package.items():
                        num_ones[w_name] += torch.sum(w_value == 1)
                        num_zeros[w_name] += torch.sum(w_value == 0)

                for w_name, w_value in local_packages[idx].items():
                    voted_results[w_name] = torch.where(num_ones > 1/3*self.num_users, torch.ones_like(w_value), voted_results[w_name]) 
                    voted_results[w_name] = torch.where(num_zeros > 1/3*self.num_users, torch.zeros_like(w_value), voted_results[w_name])

            elif self.mode == 1:
                # record the opinion of the minority 
                # minority = OrderedDict()

                accumulated_weight = WeightBuffer(local_packages[idx], "zeros")
                for user_id, package in local_packages.items():
                    accumulated_weight = WeightBuffer(package) + accumulated_weight

                accumulated_weight_state_dict = accumulated_weight.state_dict()

                for w_name, w_value in accumulated_weight_state_dict.items():
                    zeros_tensor = torch.zeros_like(w_value)
                    ones_tensor = torch.ones_like(w_value)          
                    voted_result = torch.where(w_value>0, ones_tensor, -ones_tensor)
                    voted_results[w_name] = voted_result
            
            # Generate attacker packages
            for user_id in kwargs["attacker_list"]:
                local_packages[user_id] = generate_attacker_packages(voted_results)

            if self.total_votes is None:
                self.total_votes = 0
                for w_name, w_value in local_packages[idx].items():
                    self.total_votes += w_value.numel()

            for user_id, package in local_packages.items():
                num_correct_votes = 0
                for w_name, w_values in package.items():
                    num_correct_votes += (package[w_name] == voted_results[w_name]).sum().item()
                self.user_credibility[user_id] = num_correct_votes/self.total_votes
            
            # exponential moving average and normalize the voting weight
            self.user_credibility = self.beta*self.user_previous_credibility + (1-self.beta)*self.user_credibility
            # self.vote_weight = self.user_credibility/np.sum(self.user_credibility)
            self.user_previous_credibility = self.user_credibility.copy()

            mean_credibility = np.mean(self.user_credibility)
            credibility = np.where(self.user_credibility<mean_credibility, 0, self.user_previous_credibility)
            self.vote_weight = credibility/np.sum(credibility)

            # credibility = self.user_credibility
            # self.vote_weight = credibility/np.sum(credibility)

            print(self.user_credibility[kwargs["attacker_list"]])
            print(self.vote_weight[kwargs["attacker_list"]])

            accumulated_weight = WeightBuffer(local_packages[idx], "zeros")
            for user_id, package in local_packages.items():
                accumulated_weight = WeightBuffer(package)*(self.vote_weight[user_id]) + accumulated_weight

            accumulated_weight_state_dict = accumulated_weight.state_dict()
            for w_name, w_value in accumulated_weight_state_dict.items():
                w_value.clamp_(-1+1.e-4, 1-1.e-4)
                self.global_latent_param[w_name + ".latent_param"].data = (0.5/self.k)*torch.log((1+w_value)/(1-w_value))
            
            model.load_latent_param_dict(self.global_latent_param)
        elif self.mode == 2:
            self.global_weight = accumulated_weight.state_dict()
            model.load_state_dict(self.global_weight)

def generate_attacker_packages(voted_results):
    attacker_package = OrderedDict()
    for w_name, w_value in voted_results.items():
        attacker_package[w_name] = -w_value

    return attacker_package