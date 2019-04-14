# -*- coding: utf-8 -*-

import numpy as np
import cv2, os
from PIL import Image
from cartoon import SAMPLE_IMG
from tqdm import tqdm


def load_net_in(img_fname=SAMPLE_IMG, desired_size=256):
    input_image = Image.open(img_fname).convert("RGB")
    input_image = input_image.resize((desired_size, desired_size), Image.BICUBIC)
    input_image = np.asarray(input_image)

    # preprocess, (-1, 1)
    input_image = preprocess(input_image)
    return input_image

def preprocess(xs):
    """
    # Args
        xs : rgb-ordered, [0, 255]-ranged
    # Return
        xs : bgr-ordered, [-1, +1]-ranged
    """
    # preprocess, (-1, 1)
    # rgb -> bgr
    xs = xs[...,::-1]
    xs = xs/255
    xs = -1 + 2*xs
    return xs
    
def postprocess(ys):
    """
    # Args
        ys : bgr-ordered, [-1, +1]-ranged
    # Return
        ys : rgb-ordered, [0, +1]-ranged
    """
    # bgr -> rgb
    ys = ys[...,::-1]
    # [0, 1]-range
    ys = ys * 0.5 + 0.5
    return ys

def create_smooth_dataset(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    for n, f in tqdm(enumerate(file_list)):
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.imread(os.path.join(root, f), 0)
        rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        cv2.imwrite(os.path.join(save, str(n) + '.jpg'), gauss_img)

