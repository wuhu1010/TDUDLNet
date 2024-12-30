from typing import Dict, List
import torch.utils.data as data
import torch, cv2
import time
import os
import logging
from torchsummary import summary
from glob import glob
from prettytable import PrettyTable
from torch import cuda
import numpy as np
import random
import copy
from thop import profile
from scipy import linalg
from collections import OrderedDict
import Net.denoise_net as net
from utils.dataset_admm import get_data
import utils.utils_option as option
from utils.dataset_admm import dataset_admm_denose
import utils.utils_image as image
from utils import utils_logger

if __name__ == '__main__':

    gpus = ','.join([str(i) for i in [0, 1, 2, 3]])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print("\n\nLet's use", torch.cuda.device_count(), "GPU!\n\n")
    seed_=1234
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    cuda.manual_seed_all(seed_)
    # ------------------------
    #     option_setting
    # ------------------------
    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)
    # logger
    logger_name = 'test'+time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # -------------------------
    #         dataset
    # ------------------------
    test_set = get_data(opt, 'test')


    test_loaders: List[data.DataLoader[dataset_admm_denose]] = []
    for valid in test_set:
        test_loaders.append(data.DataLoader(dataset=valid, batch_size=1, shuffle=False, num_workers=4, drop_last=True, pin_memory=True))

    # -------------------------
    #         model
    # ------------------------

    model = net.denoise_Net_admm_restormer(opt)
    pretained_path = opt["pretained_path"]
    state = torch.load(pretained_path)
    model.load_state_dict(state['state_dict'], strict=True)

    model.cuda()
    model.eval()
    #logger.info(summary(model.to('cuda')))

    # test
    avg_psnrs: Dict[str, List[float]] = {}
    avg_ssims: Dict[str, List[float]] = {}
    tags = []
    names = []
    start = 0
    end = 0
    for name in sorted(glob(os.path.join(opt['test']['dataroot_H'], '*'))):
        names.append(os.path.basename(name))
    count_ = 0
    sigma_size = len(opt['test']['sigma'])
    for test_loader in test_loaders:
        avg_psnr = 0.
        avg_ssim = 0.
        batch= 0
        with torch.no_grad():
            for batch_idx, (img_H, img_L, noise_level) in enumerate(test_loader):
                batch += 1
                img_H = img_H.cuda()
                img_L = img_L.cuda()
                noise_level = noise_level.cuda()
                # #############flops########################
                flops=None
                params =None

                start = time.time()

                test_out, aaa= model(img_L, noise_level)
                test_out = test_out
                end = time.time()
                psnr_ = image.calculate_psnr(image.tensor2uint(test_out), image.tensor2uint(img_H))
                ssim_ = image.calculate_ssim(image.tensor2uint(test_out), image.tensor2uint(img_H))
                avg_psnr = avg_psnr + psnr_
                avg_ssim = avg_ssim + ssim_
        avg_psnr = round(avg_psnr / len(test_loader), 2)
        avg_ssim = round(avg_ssim * 100 / len(test_loader), 2)
        count = count_//sigma_size
        if names[count] in avg_psnrs:
            avg_psnrs[names[count]].append(avg_psnr)
            avg_ssims[names[count]].append(avg_ssim)
        else:
            avg_psnrs[names[count]] = [avg_psnr]
            avg_ssims[names[count]] = [avg_ssim]

        count_ += 1

    logger.info('inference time：{} s'.format((end - start)))
    header = ['Dataset'] + list(opt['test']['sigma'])
    t = PrettyTable(header)
    for key, value in avg_psnrs.items():
        t.add_row([key] + value)
    logger.info(f"Test PSNR:\n{t}")

    t = PrettyTable(header)
    for key, value in avg_ssims.items():
        t.add_row([key] + value)
    logger.info(f"Test SSIM:\n{t}")

