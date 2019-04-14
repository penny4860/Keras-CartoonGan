# -*- coding: utf-8 -*-

from cartoon.utils import create_smooth_dataset
if __name__ == '__main__':
    img_root = "../../dataset/cartoon_dataset/cartoon"
    result_root = "../../dataset/cartoon_dataset/cartoon_smooth"
    create_smooth_dataset(img_root, result_root)
