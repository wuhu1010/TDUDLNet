import math
import os
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
"""
--------------------------------------------
Hongyi Zheng (github: https://github.com/natezhenghy)
07/Apr/2021
--------------------------------------------
Kai Zhang (github: https://github.com/cszn)
03/Mar/2019
--------------------------------------------
https://github.com/twhui/SRGAN-pyTorch
https://github.com/xinntao/BasicSR
--------------------------------------------
"""

##############
# path utils_ #
##############

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tif'
]


def is_img(filename: str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_img_paths(dataroot: np.str0) -> List[str]:
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_img_paths_from_root(dataroot))
    return paths


def _get_img_paths_from_root(path: str) -> List[str]:
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images: List[str] = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_img(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def makedirs(paths: Union[str, List[str]]):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


###############
# image utils_ #
###############


def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img


def imsave(img: np.ndarray, img_path: str):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def uint2single(img: np.ndarray) -> np.ndarray:
    return np.float32(img / 255.)


def uint2tensor3(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(
        2, 0, 1).float().div(255.)
    return tensor


def tensor2uint(img: torch.Tensor) -> np.ndarray:
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def single2tensor3(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def save_d(d: np.ndarray, path: str = ''):
    def merge_images(image_batch: np.ndarray):
        """
            d: C_out, C_in, d_size, d_size
        """
        h, w = image_batch.shape[-2], image_batch.shape[-1]
        img = np.zeros((int(h * 8 + 7), int(w * 8 + 7)))
        for idx, im in enumerate(image_batch):
            i = idx % 8 * (h + 1)
            j = idx // 8 * (w + 1)

            img[j:j + h, i:i + w] = im
        img = cv2.resize(img,
                         dsize=(256, 256),
                         interpolation=cv2.INTER_NEAREST)
        return img

    d = np.where(d > np.quantile(d, 0.75), 0, d)
    d = np.where(d < np.quantile(d, 0.25), 0, d)

    im_merged = merge_images(d)
    im_merged = np.absolute(im_merged)
    plt.imsave(path,
               im_merged,
               cmap='Greys',
               vmin=im_merged.min(),
               vmax=im_merged.max())


######################
# augmentation utils_ #
######################
def augment_img(img: np.ndarray, mode: int = 0) -> np.ndarray:  # 旋转与翻转的组合
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        raise ValueError


#################
# overlap-patch #
#################
def extract_ordered_patch(imgs, patch_size: tuple, stride_size: tuple):
    assert imgs.ndim > 2
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    b, c, h, w = imgs.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride_size
    # 切片数
    assert (h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0
    n_patch_h = (h - patch_h) // stride_h + 1
    n_patch_w = (w - patch_w) // stride_w + 1
    each_patch = n_patch_h * n_patch_w
    all_patch = b * each_patch
    # 预设分块大小
    patches = torch.zeros(all_patch, c, patch_h, patch_w)
    patch_idx = 0

    # 依次切块
    for img in imgs:
        for i in range(n_patch_h):
            for j in range(n_patch_w):
                x1 = j * stride_w
                x2 = x1 + patch_w
                y1 = i * stride_h
                y2 = y1 + patch_h
                patches[patch_idx] = img[:, y1:y2, x1:x2]
                patch_idx += 1

    return patches


def rebuild_patches(patches, img_size: tuple, stride_size: tuple):
    assert patches.ndim == 4
    img_h, img_w = img_size
    stride_h, stride_w = stride_size
    n_patches, c, patch_h, patch_w = patches.shape
    assert (img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0
    # 切片数
    n_patch_h = (img_h - patch_h) // stride_h + 1
    n_patch_w = (img_w - patch_w) // stride_w + 1
    each_patches = n_patch_h * n_patch_w
    all_patches = n_patches // each_patches
    # 预设重建图
    imgs = torch.zeros(all_patches, c, img_h, img_w)
    weights = torch.zeros_like(imgs)

    # 依次重建
    for img_idx, (img, weight) in enumerate(zip(imgs, weights)):
        start = img_idx * each_patches

        for i in range(n_patch_h):
            for j in range(n_patch_w):
                x1 = j * stride_w
                x2 = x1 + patch_w
                y1 = i * stride_h
                y2 = y1 + patch_h
                patch_idx = start + i * n_patch_h + j
                img[:, y1:y2, x1:x2] = patches[patch_idx]
                weight[:, y1:y2, x1:x2] += 1
    imgs /= weights

    return imgs


###########
# metrics #
###########
def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0):
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = (img1[border:h - border, border:w - border]+ 0.03) / 1.03
    img2 = (img2[border:h - border, border:w - border]+ 0.03) / 1.03

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse: float = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray,
                   border: int = 0) -> float:
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = (img1[border:h - border, border:w - border]+ 0.03) / 1.03
    img2 = (img2[border:h - border, border:w - border]+ 0.03) / 1.03

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims: List[float] = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    s: float = ssim_map.mean()
    return s
