import os
import copy
import pickle
import logging
import time
import numpy as np

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My libraries
from config import load_config
from config.utils import *
from qnn.dataset import *
from qnn.initialize import *
from fedlearning.attacker import *
from fedlearning.myoptimizer import *
from fedlearning.validate import validate_and_log

def train(model, config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        model (nn.Module):       the model to be trained.
        config (class):          the user defined configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """    
    user_id_sets = generate_attacker_list(config)
    user_ids = user_id_sets["normal_user_ids"]
    attacker_list = user_id_sets["attacker_list"]
    print(attacker_list)

    # initialize the optimizer for the server model
    dataset = assign_user_data(config, logger)
    dataset_type = parse_dataset_type(config)


    # global_updater = GlobalUpdater(config, model.state_dict())
    global_updater = GlobalUpdater(config, model, 
                        dim=record["num_parameters"], 
                        device=config.device)

    # before optimization, report the result first
    validate_and_log(model, dataset, config, record, logger)

    for comm_round in range(config.rounds):
        # Sample a fraction of users randomly
        user_ids_candidates = user_ids
        
        # Wait for all users updating locally
        local_packages = {}
        for i, user_id in enumerate(user_ids):
            user_resource = assign_user_resource(config, user_id, 
                                dataset["train_data"], dataset["user_with_data"])
            updater = LocalUpdater(user_resource, config)
            updater.local_step(model, test_data=dataset["test_data"], config=config)
            # updater.local_step(model, test_data=dataset["test_data"], config=config)


            if config.save:
                torch.save(updater.local_weight, "./models/bnn_{:d}_{:d}.pth".format(comm_round, user_id))

            local_package = updater.uplink_transmit()
            local_packages[user_id] = local_package

        # Update the global model
        # global_updater.global_step(model, local_packages, record=record)
        global_updater.global_step(model, local_packages, attacker_list=attacker_list, record=record)

        if config.save:
            torch.save(model.latent_param_dict(), "./models/gbnn_{:d}.pth".format(comm_round))

        # Validate the model performance and log
        logger.info("Round {:d}".format(comm_round))
        validate_and_log(model, dataset, config, record, logger)

        if comm_round == config.scheduler[0]:
            config.scheduler.pop(0)


    if config.mode == 0 or config.mode == 1:
        torch.save(model.latent_param_dict(), "vote.pth")
    elif config.mode == 2:
        torch.save(model.state_dict(), "FedAvg.pth")

def main():
    config = load_config()
    logger = init_logger(config)

    # ----------------------------
    # mode 0: train the local ternary neural network 
    # mode 2: train the full precision model with FedAvg
    #-----------------------------
    if config.mode == 0 or config.mode == 1:
        model = init_qnn_model(config, logger)
    elif config.mode == 2:
        model = init_full_model(config, logger)
    else:
        raise NotImplementedError("invalid mode")

    record = init_record(config, model)

    start = time.time()
    train(model, config, logger, record)
    end = time.time()

    save_record(config, record)
    logger.info("{:.3} mins has elapsed".format((end-start)/60))

if __name__ == "__main__":
    main()

