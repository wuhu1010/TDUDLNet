import json
import os
from datetime import datetime
from typing import Any, Dict

import commentjson
'''
# --------------------------------------------
# Hongyi Zheng (github: https://github.com/natezhenghy)
# 07/Apr/2021
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
`
'''


def get_timestamp() -> str:
    return datetime.now().strftime('_%y%m%d_%H%M%S')


def parse(opt_path: str, is_train: bool = True) -> Dict[str, Any]:

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    with open(opt_path) as file:
        opt: Dict[str, Any] = commentjson.load(file)

    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # data
    # ----------------------------------------
    if 'scale' not in opt:
        opt['scale'] = 1

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


'''
# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
'''


def save(opt: Dict[str, Any]):
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    _, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename + get_timestamp() + ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


'''
# --------------------------------------------
# dict to string for logger
# --------------------------------------------
'''


def dict2str(opt: Dict[str, Any], indent_l: int = 1):
    msg: str = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
