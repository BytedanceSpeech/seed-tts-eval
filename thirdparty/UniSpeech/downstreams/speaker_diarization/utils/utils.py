import os
import struct
import logging
import torch
import math
import numpy as np
import random
import yaml
import torch.distributed as dist
import torch.nn.functional as F


# ------------------------------ Logger ------------------------------
# log to console or a file
def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# log to concole and file at the same time
def get_logger_2(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(name)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter(fmt=format_str, datefmt=date_format)
    f_format = logging.Formatter(fmt=format_str, datefmt=date_format)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


# ------------------------------ Logger ------------------------------

# ------------------------------ Pytorch Distributed Training ------------------------------
def getoneNode():
    nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    import re
    text = re.split('[-\[\]]', nodelist)
    if ('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]


def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    dist.init_process_group("nccl", init_method=host_addr_full,
                            rank=rank, world_size=world_size)
    num_gpus = torch.cuda.device_count()
    # torch.cuda.set_device(local_rank)
    assert dist.is_initialized()


def cleanup():
    dist.destroy_process_group()


def average_gradients(model, world_size):
    size = float(world_size)
    for param in model.parameters():
        if (param.requires_grad and param.grad is not None):
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def data_reduce(data):
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    return data / torch.distributed.get_world_size()


# ------------------------------ Pytorch Distributed Training ------------------------------


# ------------------------------ Hyper-parameter Dynamic Change ------------------------------
def reduce_lr(optimizer, initial_lr, final_lr, current_iter, max_iter, coeff=1.0):
    current_lr = coeff * math.exp((current_iter / max_iter) * math.log(final_lr / initial_lr)) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def get_reduce_lr(initial_lr, final_lr, current_iter, max_iter):
    current_lr = math.exp((current_iter / max_iter) * math.log(final_lr / initial_lr)) * initial_lr
    return current_lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ------------------------------ Hyper-parameter Dynamic Change ------------------------------

# ---------------------- About Configuration --------------------
def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    return dict(yaml_config, **kwargs)


def store_yaml(config_file, store_path, **kwargs):
    with open(config_file, 'r') as f:
        config_lines = f.readlines()

    keys_list = list(kwargs.keys())
    with open(store_path, 'w') as f:
        for line in config_lines:
            if ':' in line and line.split(':')[0] in keys_list:
                key = line.split(':')[0]
                line = '{}: {}\n'.format(key, kwargs[key])
            f.write(line)


# ---------------------- About Configuration --------------------


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def set_seed(seed=66):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# when store the model wrongly with "module" involved,
# we remove it here
def correct_key(state_dict):
    keys = list(state_dict.keys())
    if 'module' not in keys[0]:
        return state_dict
    else:
        new_state_dict = {}
        for key in keys:
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
        return new_state_dict


def validate_path(dir_name):
    """
    :param dir_name: Create the directory if it doesn't exist
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name != ''):
        os.makedirs(dir_name)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
