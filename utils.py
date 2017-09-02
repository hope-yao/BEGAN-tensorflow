from __future__ import print_function

import os
import math
import json
import logging
from PIL import Image
from datetime import datetime
import numpy as np
def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=4,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.ones([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8) *127
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=4,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
    return im

import tensorflow as tf
def list2tensor(alist,dim=0):
    for i, list_i in enumerate(alist):
        if i == 0:
            atensor = list_i
        else:
            atensor = tf.concat([atensor, list_i], dim)
    return atensor


import dateutil.tz

def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import numpy as np
# import seaborn
# def new_save_image(tensor, filename, nrow=8, padding=4,
#                normalize=False, scale_each=False):
#     ndarr = make_grid(tensor, nrow=nrow, padding=padding,
#                             normalize=normalize, scale_each=scale_each)
#     ndarr = np.asarray(ndarr,"float32")
#     fig = plt.figure(figsize=(20,15))
#     ax = plt.gca()
#     im = ax.imshow(ndarr / np.max(ndarr))
#     # create an axes on the right side of ax. The width of cax will be 5%
#     # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(im, cax=cax,cmap="hot")#
#     plt.savefig(filename)
#     # plt.show()
#     plt.close(fig)