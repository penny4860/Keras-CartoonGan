# -*- coding: utf-8 -*-

from cartoon import MODEL_ROOT
from cartoon.models import cartoon_generator, postprocess
from cartoon.utils import load_net_in
import os
import numpy as np
import matplotlib.pyplot as plt

INPUT_IMG_FNAME = "sample_in/sjtu.jpg"
INPUT_SIZE = 512

if __name__ == '__main__':
    model_names = ["params/Hayao.h5", "params/Hosoda.h5", "params/Paprika.h5", "params/Shinkai.h5"]
    
    model = cartoon_generator(input_size=INPUT_SIZE)

    fig, ax = plt.subplots()
    plt.subplot(1, 5, 1)
    plt.axis('off')
    plt.title("input photo")
    plt.imshow(postprocess(load_net_in(INPUT_IMG_FNAME)))

    for j, model_path in enumerate(model_names):
        model.load_weights(model_path)
    
        imgs = np.expand_dims(load_net_in(INPUT_IMG_FNAME, desired_size=INPUT_SIZE), axis=0)
        ys = model.predict(imgs)
        y = postprocess(ys)[0]

        plt.subplot(1, 5, j+2)
        plt.axis('off')
        plt.title(os.path.basename(model_path))
        plt.imshow(y)
    plt.subplots_adjust(bottom = 0, top = 1, hspace = 0.01, wspace = 0.01)
    plt.show()
