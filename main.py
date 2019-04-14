# -*- coding: utf-8 -*-

from cartoon import MODEL_ROOT
from cartoon.models import cartoon_generator
from cartoon.utils import postprocess
from cartoon.utils import load_net_in
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

INPUT_IMG_FNAME = "sample_in/sjtu.jpg"
INPUT_SIZE = 512

argparser = argparse.ArgumentParser(description='CartoonGan')
argparser.add_argument(
    '-i',
    '--input_img_fname',
    default=INPUT_IMG_FNAME,
    help='input image filename')
argparser.add_argument(
    '-s',
    '--input_size',
    default=INPUT_SIZE,
    help='generator network input size')


if __name__ == '__main__':
    args = argparser.parse_args()
    model_names = ["params/Hayao.h5", "params/Hosoda.h5", "params/Paprika.h5", "params/Shinkai.h5"]
    model = cartoon_generator(input_size=args.input_size)

    fig, ax = plt.subplots()
    plt.subplot(1, 5, 1)
    plt.axis('off')
    plt.title("input photo")
    plt.imshow(postprocess(load_net_in(args.input_img_fname)))

    for j, model_path in enumerate(model_names):
        model.load_weights(model_path)
    
        imgs = np.expand_dims(load_net_in(args.input_img_fname, desired_size=args.input_size), axis=0)
        ys = model.predict(imgs)
        y = postprocess(ys)[0]

        plt.subplot(1, 5, j+2)
        plt.axis('off')
        plt.title(os.path.basename(model_path).split(".")[0])
        plt.imshow(y)
    plt.show()
