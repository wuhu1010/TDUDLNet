from typing import List
import torch.utils.data as data
import torch
import time
import os
from tqdm import tqdm
import logging
from torchsummary import summary
from torch import cuda, optim
import numpy as np
import random
import matplotlib.pyplot as plt
from math import log

from Net.DGUNet import DGUNet
import Net.denoise_net as net
from Net.restormer_arch import Restormer
from utils.dataset_admm import get_data
from utils.loss_function import loss_function
import utils.utils_option as option
from utils.dataset_admm import dataset_admm_denose
import utils.utils_image as image
from utils import utils_logger


def adjust_learning_rate(opt, epo, lr_ini, max_epoch):
    """Sets the learning rate to the initial LR decayed by 5 every 50 epochs"""
    P1=50
    P2=200-P1
    if epo<P1:
        lr = lr_ini * (0.65 ** (epo // (P1//log(0.1, 0.65))))
    else:
        lr = lr_ini * 0.1 * (0.85 ** ((epo-P1) // (P2 // log(0.1, 0.85))))

    for param_group in opt.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    gpus = ','.join([str(i) for i in [0, 1, 2, 3]])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print("\n\nLet's use", torch.cuda.device_count(), "GPU!\n\n")
        
    # ------------------------
    #     option_setting
    # ------------------------
    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)
    # logger
    logger_name = 'train'+time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # -------------------------
    #         dataset
    # ------------------------
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')

    logger.info(f"train set, {len(train_set)}, test set, {len(valid_set)}")                           # 最大12
    train_loader = data.DataLoader(dataset=train_set, batch_size=opt['batch_size'], shuffle=False, num_workers=2, pin_memory=False)
    test_loaders: List[data.DataLoader[dataset_admm_denose]] = []
    for valid in valid_set:
        test_loaders.append(data.DataLoader(dataset=valid, batch_size=1, shuffle=False, num_workers=1, drop_last=True, pin_memory=True))

    # -------------------------
    #         model
    # ------------------------
    model = net.denoise_Net_admm_restormer(opt)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=5,
                                                           verbose=False, threshold=1e-3, threshold_mode='abs',
                                                           cooldown=0, min_lr=0, eps=1e-8)
    criterion = loss_function(opt['loss_function_index'])
    logger.info(summary(model.to('cuda')))
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Number of parameter: {total / 1e6 :.2f}M")

    logger.info("start training...")

    start = time.time()
    loss_train = []
    test__loss = []
    test__psnr = []
    test__ssim = []
    best_psnr = 0
    best_epoch = 0
    batch_accumulation = 1
    max_accumulation = 0

    if opt["pretained_path"]["index"]:
        state = torch.load(opt['pretained_path']["path"])
        model.load_state_dict(state['state_dict'], strict=False)


    eval_num = 5

    for epoch in range(20, opt["max_epoch"]):
        if epoch < 200:
            adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
        else:
            if 'psnr_val_rgb' not in vars():
                psnr_val_rgb = 0
            reduce_schedule.step(psnr_val_rgb)

        loss = 0
        loss1 = 0
        for batch_idx, (img_H, img_L, noise_level) in enumerate(tqdm(train_loader,0)):

            img_H = img_H.cuda()
            img_L = img_L.cuda()
            output, preds = model(img_L, noise_level)
            # output = model(img_L)

            #########################################
            loss = criterion(output[0], img_H)  # / batch_accumulation
            loss.backward()
            if (batch_idx+1)%batch_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss1 = loss1 + torch.detach(loss)
        loss_train.append((loss1).to('cpu'))
        logger.info(f"epoch:[{epoch + 1:.0f}//{opt['max_epoch']:.0f}], 当前总loss值：{loss1:.4f}")

        #### Evaluation ####
        if (epoch+1)%eval_num==0 or epoch<5 or epoch>opt['max_epoch']-5 or epoch==151:
            model.eval()
            test_loss = 0
            test_psnr = 0
            test_ssim = 0
            batch = 0
            for test_loader in test_loaders:
                for batch_idx, (img_H, img_L, noise_level) in enumerate(tqdm(test_loader,0)):
                    with torch.no_grad():
                        batch += 1
                        img_H = img_H.cuda()
                        img_L = img_L.cuda()
                        test_out, aaa = model(img_L, noise_level)

                        test_loss = test_loss + criterion(test_out, img_H)
                        test_out_ = image.tensor2uint(test_out)

                        test_psnr = test_psnr + image.calculate_psnr(image.tensor2uint(test_out), image.tensor2uint(img_H))
                        test_ssim = test_ssim + image.calculate_ssim(image.tensor2uint(test_out), image.tensor2uint(img_H))

            psnr_val_rgb = test_psnr / batch
            if psnr_val_rgb > best_psnr:
                max_accumulation = 0
                best_psnr = psnr_val_rgb
                best_epoch = epoch+1
                # best_iter = batch_idx
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(opt['model_save'],"model_relu_gray_best.pth"))
            else:
                max_accumulation += 1

            logger.info('[epoch {:.0f}  PSNR: {:.4f} --- best_epoch {:.0f}  Best_PSNR {:.4f}]'
                        .format(epoch+1, psnr_val_rgb, best_epoch, best_psnr))

            test__loss.append((test_loss / batch).to("cpu"))
            test__psnr.append(test_psnr / batch)
            test__ssim.append(test_ssim / batch)
            logger.info('Test set other metrics: Current learning rate:{:.2e}，loss:{:.4f},ssim:{:.4f}'.format(optimizer.param_groups[0]['lr'], test_loss / batch, test_ssim / batch))

        torch.save({'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(opt['model_save'], 'model{}.pth'.format('_latest_name')))
        if max_accumulation == 5:
            break

    end = time.time()
    logger.info('Training time：{} hour'.format((end-start)/3600))
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(range(1, len(loss_train)+1), loss_train)
    plt.title("loss_train")
    plt.subplot(222)
    plt.plot(range(1, len(test__loss) + 1), test__loss)
    plt.title("test__loss")
    plt.subplot(223)
    plt.plot(range(1, len(test__psnr) + 1), test__psnr)
    plt.title("test__psnr")
    plt.subplot(224)
    plt.plot(range(1, len(test__ssim) + 1), test__ssim)
    plt.title("test__ssim")
    plt.show()

