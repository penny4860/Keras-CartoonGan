# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from cartoon import SAMPLE_IMG

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

