# -*- coding: utf-8 -*-

from cartoon.models import cartoon_discriminator
from cartoon.seq import BatchGenerator, create_callbacks
import tensorflow as tf


if __name__ == '__main__':
    import glob
    import numpy as np
    photo_fnames = glob.glob("C:/Users/penny/git/dataset/cartoon_dataset/photo/*.*")
    cartoon_fnames = glob.glob("C:/Users/penny/git/dataset/cartoon_dataset/cartoon/*.*")
    
    photo_generator = BatchGenerator(photo_fnames,
                                     batch_size=2,
                                     shuffle=True,
                                     label=0,
                                     input_size=256)
    cartoon_generator = BatchGenerator(cartoon_fnames,
                                       batch_size=2,
                                       label=1,
                                       shuffle=True,
                                       input_size=256)
     
    model = cartoon_discriminator()
    model.summary()
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001))
    
    for i in range(30):    
        xs, ys = cartoon_generator[0]
        d_loss_real = model.train_on_batch(xs, ys)
        xs, ys = photo_generator[0]
        d_loss_fake = model.train_on_batch(xs, ys)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print(i, d_loss)
    


