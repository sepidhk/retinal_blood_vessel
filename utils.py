# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import time
import random
import cv2
import torch
import numpy as np


def seed_all(seed=1142):
    """seed everything for reproducibility"""
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def create_dir(dir_name):
    """create directory"""
    os.makedirs(dir_name, exist_ok=True)
    
def computation_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    

    
    