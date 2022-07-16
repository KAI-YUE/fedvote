import os

# PyTorch Libraries
import torch

# My Libraries
from config.loadconfig import load_config
from qnn.networks import *
from qnn.qnn_struct import *

def init_qnn_model(config, logger):
    # initialize the tnn_model
    sample_size = config.sample_size[0] * config.sample_size[1]
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    qnn = qnn_registry[config.qnn_model](in_dims=sample_size*config.channels, in_channels=config.channels)
    
    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-trained full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)
    else:
        logger.info("--- Train quantized model from scratch. ---")
        full_model.apply(init_weights)

    init_qnn_latent_params(qnn, full_model)
    qnn.freeze_weight()
    qnn = qnn.to(config.device)

    if os.path.exists(config.qnn_weight_dir):
        latent_state_dict = torch.load(config.qnn_weight_dir)
        qnn.load_latent_param_dict(latent_state_dict)

    return qnn

def init_full_model(config, logger):
    logger.info("Initialize the full precision model.")
    # initialize the full precision model
    sample_size = config.sample_size[0] * config.sample_size[1] 
    full_model = nn_registry[config.full_model](in_dims=sample_size*config.channels, in_channels=config.channels, out_dims=config.classes)
    full_model.apply(init_weights)
    
    if config.freeze_fc:
        full_model.freeze_final_layer()

    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-trained full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)

    full_model.to(config.device)

    return full_model


def init_qnn_latent_params(model, ref_model, **kwargs):
    """Initialize real valued latent parameters.

    Args:
        ref_model (nn.Module):     the reference floating point model
    """
    ref_state_dict = ref_model.state_dict()
    model.load_state_dict(ref_state_dict)
    named_modules = model.named_modules()
    next(named_modules)

    config = load_config()
    k = config.k

    for module_name, module in named_modules:
        if not hasattr(module, "weight"):
            continue
        elif not hasattr(module.weight, "latent_param"):
            continue
            
        # normalize the weight
        ref_w = ref_state_dict[module_name + ".weight"]
        ref_w = ref_w/ref_w.std()
        ref_w.clamp_(-1+1.e-3, 1-1.e-3)

        # k * latent_weight = tanh^{-1}(ref_w)
        module.weight.latent_param.data = (1/k)*0.5*torch.log((1 + ref_w)/(1 - ref_w))
    

nn_registry = {
    "NaiveCNN":         NaiveCNN,
    "bn":               FullPrecision_BN,
    "lenet":            LeNet_5,
    "vgg":              VGG_7,
}

qnn_registry = {
    "qnn":              QuantizedNeuralNet_BN,
    "qnn_lenet":        QuantizedLeNet_BN,
    "qnn_vgg":          QuantizedVGG_7
}
