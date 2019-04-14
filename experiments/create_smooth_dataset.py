# -*- coding: utf-8 -*-

import argparse
from cartoon.utils import create_smooth_dataset

INPUT_IMG_ROOT = "../../dataset/cartoon_dataset/cartoon"
OUTPUT_IMG_ROOT = "../../dataset/cartoon_dataset/cartoon"

argparser = argparse.ArgumentParser(description='Create Smooth Dataset')
argparser.add_argument(
    '-i',
    '--input_img_root',
    default=INPUT_IMG_ROOT,
    help='input image root')
argparser.add_argument(
    '-o',
    '--output_img_root',
    default=OUTPUT_IMG_ROOT,
    help='output_img_root')


if __name__ == '__main__':
    args = argparser.parse_args()
    create_smooth_dataset(args.input_img_root, args.output_img_root)

