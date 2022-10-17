import torch
import getpass
import logging
import os
import yaml
import warnings
import collections
from pathlib import Path
from itertools import product
from importlib import import_module
from colorlog import ColoredFormatter


# Logging
# =======

def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


# General utils
# =============


# Path utils
# ==========

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_project_root():
    return str(Path(__file__).parent.parent.parent)


# MultiGPU
# ========

class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Data
# ====

def normalization_params():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (mean, std)

def to_device(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict


# Config
# ======

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.
    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.
    Returns:
        list[module] | module | None: The imported modules.
    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def split_dict(d):
    to_product = []  # [[('a', 1), ('a', 2)], [('b', 3),], ...]
    for k, v in d.items():
        if isinstance(v, list):
            to_product.append([(k, i) for i in v])
        elif isinstance(v, dict):
            to_product.append([(k, i) for i in split_dict(v)])
        else:
            to_product.append([(k, v)])
    return [dict(l) for l in product(*to_product)]

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dict_to_str(d, sep='_'):
    flattened_d = flatten_dict(d, sep='.')
    string = ''
    for key, val in flattened_d.items():
        string += key + sep + str(val) + sep
    return string[:-1]
