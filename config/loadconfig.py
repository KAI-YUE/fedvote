import yaml
import json
import os
import numpy as np

def load_config(filename=None):
    """Load configurations of yaml file"""
    current_path = os.path.dirname(__file__)

    if filename is None:
        filename = "config.yaml"

    with open(os.path.join(current_path, filename), "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # Empty class for yaml loading
    class cfg: pass
    
    for key in config:
        setattr(cfg, key, config[key])

    cfg.scheduler.append(np.nan)

    return cfg
