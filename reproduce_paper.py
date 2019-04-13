# -*- coding: utf-8 -*-

from cartoon.models import cartoon_generator
from cartoon.utils import load_net_in
import numpy as np
import matplotlib.pyplot as plt
from cartoon.utils import postprocess

if __name__ == '__main__':
    input_size = 512
    img_files = ["sample_in/in1.png", "sample_in/in2.png", "sample_in/in3.png"]
    model_names = ["params/Shinkai.h5", "params/Hayao.h5"]
    n_rows = n_cols = 3
    
    model = cartoon_generator(input_size=input_size)

    fig, ax = plt.subplots()
    for i, fname in enumerate(img_files):
        plt.subplot(n_rows, n_cols, 3*i + 1)
        plt.axis('off')
        # plt.title("input photo")
        plt.imshow(postprocess(load_net_in(fname)))
    
        for j, model_path in enumerate(model_names):
            model.load_weights(model_path)
        
            imgs = np.expand_dims(load_net_in(fname, desired_size=input_size), axis=0)
            ys = model.predict(imgs)
            y = postprocess(ys)[0]
    
            plt.subplot(n_rows, n_cols, 3*i+j+2)
            plt.axis('off')
            # plt.title(os.path.basename(model_path))
            plt.imshow(y)
    plt.subplots_adjust(bottom = 0, top = 1, hspace = 0.01, wspace = 0.01)
    plt.show()
