import numpy as np

# PyTorch Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 

# My Libraries
from qnn.dataset import UserDataset
from qnn import nn_registry
from config.utils import parse_dataset_type

def validate_and_log(model, dataset, config, record, logger):
    dataset_type = parse_dataset_type(config)

    with torch.no_grad():
        model.eval()
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)
        
        record["loss"].append(loss)
        record["testing_accuracy"].append(test_acc)

        if config.mode == 0:
            sampled_tnn_acc = test_tnn_accuracy(model, dataset["test_data"], config)
            record["sampled_qnn_accuracy"].append(sampled_tnn_acc)

            logger.info("Sampled qnn test accuracy {:.4f}".format(sampled_tnn_acc))
        
        elif config.mode == 1:
            sampled_bnn_acc = test_bnn_accuracy(model, dataset["test_data"], config)
            record["sampled_qnn_accuracy"].append(sampled_bnn_acc)

            logger.info("Sampled qnn test accuracy {:.4f}".format(sampled_bnn_acc))

        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))
        logger.info("")

        model.train()

def test_accuracy(model, test_dataset, dataset_type, device="cuda"):
    with torch.no_grad():
        model.eval()
        dataset = UserDataset(test_dataset["images"], test_dataset["labels"], dataset_type)
        num_samples = test_dataset["labels"].shape[0]
        predicted_labels = np.zeros_like(test_dataset["labels"])
        accuracy = 0

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        testing_data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for i, samples in enumerate(testing_data_loader):
            results = model(samples["image"].to(device))
            predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
            accuracy += np.sum(predicted_labels == test_dataset["labels"][i*batch_size: (i+1)*batch_size]) / results.shape[0]
        
        accuracy /= dividers
        model.train()

    return accuracy

def test_bnn_accuracy(bnn_model, test_dataset, config):
    with torch.no_grad():
        bnn_model.eval()
        sample_size = config.sample_size[0] * config.sample_size[1]
        sampled_bnn = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
        sampled_bnn.load_state_dict(bnn_model.state_dict())
        
        sampled_bnn_state_dict = sampled_bnn.state_dict()

        entropy = 0

        named_modules = bnn_model.named_modules()
        next(named_modules)
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            ones_tensor = torch.ones_like(module.weight.latent_param)

            # max likelihood
            # sampled_bnn_state_dict[module_name + ".weight"] = module.weight.latent_param.sign()
            sampled_bnn_state_dict[module_name + ".weight"] = torch.where(module.weight.latent_param.data>0, ones_tensor, -ones_tensor)

            # rand_variable = torch.rand_like(module.weight.latent_param)
            # prob_equal_one = 0.5*(torch.tanh(config.k*module.weight.latent_param) + 1)
            # ones_tensor = torch.ones_like(module.weight.latent_param)
            # zeros_tensor = torch.zeros_like(module.weight.latent_param)
            # sampled_bnn_state_dict[module_name + ".weight"] = torch.where(rand_variable < prob_equal_one, ones_tensor, -ones_tensor)

        sampled_bnn.load_state_dict(sampled_bnn_state_dict)
        sampled_bnn = sampled_bnn.to(config.device)
        dataset_type = parse_dataset_type(config)
        acc = test_accuracy(sampled_bnn, test_dataset, dataset_type, config.device)

        bnn_model.train()

    return acc     

def test_tnn_accuracy(qnn_model, test_dataset, config):
    """test the accuracy of a sampled qnn from the latent distribution.
    """
    with torch.no_grad():
        qnn_model.eval()
        sample_size = config.sample_size[0] * config.sample_size[1]
        sampled_qnn = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
        sampled_qnn.load_state_dict(qnn_model.state_dict())
        
        sampled_qnn_state_dict = sampled_qnn.state_dict()

        named_modules = qnn_model.named_modules()
        next(named_modules)
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif not hasattr(module.weight, "latent_param"):
                continue
            
            # probability vector [pr(w_b=-1), pr(w_b=0), pr(w_b=1)] 
            pseudo_weight = torch.tanh(config.k*module.weight.latent_param)
            idx_m1 = (pseudo_weight < 0)
            idx_p1 = (pseudo_weight > 0)
            
            weight_shape = torch.tensor(pseudo_weight.shape).tolist()
            weight_shape.append(3)
            prob = torch.zeros(weight_shape)
            prob = prob.to(pseudo_weight.device)

            prob[..., 0][idx_m1] = pseudo_weight[idx_m1].abs()
            prob[..., 2][idx_p1] = pseudo_weight[idx_p1].abs()
            prob[..., 1] = 1 - prob[..., 0] - prob[..., 2] 
            
            # max likelihood sampling
            sampled_qnn_state_dict[module_name + ".weight"] = torch.argmax(prob, dim=-1) - 1 
            
            # random sampling
            # ones_tensor = torch.ones_like(pseudo_weight)
            # random_variable = torch.rand_like(pseudo_weight)
            # sampled_weight = torch.zeros_like(pseudo_weight)
            # sampled_weight = torch.where(random_variable < prob[..., 0], -ones_tensor, sampled_weight)
            # sampled_weight = torch.where(1 - prob[...,2] < random_variable, ones_tensor, sampled_weight)
            # sampled_qnn_state_dict[module_name + ".weight"] = sampled_weight

        sampled_qnn.load_state_dict(sampled_qnn_state_dict)
        sampled_qnn = sampled_qnn.to(config.device)
        dataset_type = parse_dataset_type(config)
        acc = test_accuracy(sampled_qnn, test_dataset, dataset_type, config.device)

        qnn_model.train()
    return acc


def train_loss(model, train_dataset, dataset_type, device="cuda"):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        dataset = UserDataset(train_dataset["images"], train_dataset["labels"], dataset_type)
        loss = torch.tensor(0.).to(device)

        dividers = 100
        batch_size = int(len(dataset)/dividers)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        counter = 0
        for samples in data_loader:
            results = model(samples["image"].to(device))
            loss += criterion(results, samples["label"].to(device))
        
        loss /= dividers

    return loss.item()