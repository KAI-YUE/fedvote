import os
import logging
import numpy as np
import pickle
import datetime

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def init_logger(config):
    """Initialize a logger object. 
    """
    log_level = config.log_level
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    fh = logging.FileHandler(config.log_file)
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    return logger

def parse_dataset_type(config):
    if "fmnist" in config.train_data_dir:
        type_ = "fmnist"
    elif "mnist" in config.train_data_dir:
        type_ = "mnist"
    elif "cifar" in config.train_data_dir:
        type_ = "cifar"
    
    return type_

def save_record(config, record):
    current_path = os.path.dirname(__file__)
    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time ,'%H_%M')
    file_name = config.record_dir.format(current_time_str)
    with open(os.path.join(current_path, file_name), "wb") as fp:
        pickle.dump(record, fp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_record(config, model):
    record = {}
    # number of trainable parameters
    record["num_parameters"] = count_parameters(model)

    # put some config info into record
    record["tau"] = config.tau
    record["batch_size"] = config.local_batch_size
    record["lr"] = config.lr
    record["momentum"] = config.momentum
    record["weight_decay"] = config.weight_decay

    # initialize data record 
    record["testing_accuracy"] = []
    record["sampled_qnn_accuracy"] = []
    record["loss"] = []

    return record
