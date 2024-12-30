import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_gradient_magnitude


class loss_function(torch.nn.Module):
    def __init__(self, l, eps=1e-3):
        super(loss_function, self).__init__()
        self.l = l
        self.eps = eps
        self.mseloss = nn.MSELoss()
        self.L1loss = nn.L1Loss()

    def forward(self, output, Y_gt, x= None):
        l_ = 0
        if self.l == 1:
            l_ = self.L1loss(output, Y_gt)  # 差值绝对值的平均值DCDicl  psnr 26.5296 epoch 80
        elif self.l == 2:
            l_ = torch.norm(output - Y_gt, 'fro') / torch.norm(Y_gt, 'fro')  # F范数  ADMMpsnr 26.0434
        elif self.l == 3:
            l_ = self.mseloss(output, Y_gt)  # 平方差的平均值  psnr 25.9795 epoch 80
        elif self.l == 4:
            diff = output - Y_gt
            l_ = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))  # 平方根的平均值（和1一样）
        elif self.l == 5:
            dx1, dy1 = gradient(x)
            dx2, dy2 = gradient(Y_gt)
            grad_diff_x = torch.norm(torch.exp(-1000 * dx2) * dx1, p=1)
            grad_diff_y = torch.norm(torch.exp(-1000 * dy2) * dy1, p=1)           # 不太行，不知道是计算方式不对还是怎样
            l_ = self.L1loss(output, Y_gt) + 1e-4*(grad_diff_x + grad_diff_y)     # 在一范数的基础上加了个系数表示或输出图像的平滑性（通过梯度体现）
        loss = l_
        return loss


def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy