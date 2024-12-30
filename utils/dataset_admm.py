import torch.utils.data as data
import os
import torch
from typing import List
import numpy as np
from utils import utils_image as util
import random
from copy import deepcopy
from glob import glob
# Z = scipy.linalg.solve_sylvester(a, b, q)


class dataset_admm_denose(data.Dataset):

    def __init__(self, opt, task):

        self.opt = opt
        self.task = task
        self.img_paths = util.get_img_paths(self.opt['dataroot_H'])

        self.sigma = opt['sigma']
        self.n_channels = opt['n_channels']
        if 'H_size' in opt:
            self.patch_size = opt['H_size']

    def __getitem__(self, index):
        # get H image
        img_path = self.img_paths[index]
        img_H = util.imread_uint(img_path, self.n_channels)

        H, W = img_H.shape[:2]
        
        if self.task == 'train':

            # crop(随机裁剪path_size*path_size大小图片)
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation（随机通过旋转等操作进行数据强化）
            patch_H = util.augment_img(patch_H, mode=np.random.randint(0, 8))

            # HWC to CHW, numpy(uint) to tensor
            img_H = util.uint2tensor3(patch_H)
            img_L: torch.Tensor = img_H.clone()

            # get noise level
            noise_level: torch.FloatTensor = torch.FloatTensor(
                [np.random.uniform(self.sigma[0], self.sigma[1])]) / 255.0

            # add noise
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)  # torch.Tensor()

            # add noise
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma / 255.0, img_L.shape)

            noise_level = torch.FloatTensor([self.sigma / 255.0])

            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            h, w = img_H.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, (w // 8 * 8))
            img_H = img_H[..., top, left]
            img_L = img_L[..., top, left]


        return img_H, img_L, noise_level

    def __len__(self):
        return len(self.img_paths)


def get_data(opt, task):
    if task == 'train':
        opt_ = opt[task]
        dataset = dataset_admm_denose(opt_, task)

        return dataset
    else:
        datasets: List[dataset_admm_denose] = []
        opt_ = opt[task]
        paths = glob(os.path.join(opt_['dataroot_H'], '*'))
        sigmas = opt_['sigma']

        opt_dataset_sub = deepcopy(opt_)
        for path in sorted(paths):
            for sigma in sigmas:
                opt_dataset_sub['dataroot_H'] = path
                opt_dataset_sub['sigma'] = sigma
                datasets.append(dataset_admm_denose(opt_dataset_sub, task))

        return datasets
