# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
TensorBoard = tf.keras.callbacks.TensorBoard
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
Sequence = tf.keras.utils.Sequence


def create_callbacks(saved_weights_name="mobile_encoder.h5"):
    checkpoint = ModelCheckpoint(saved_weights_name, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)
    callbacks = [checkpoint]
    callbacks.append(ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.5,
        patience   = 10,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-5
    ))
    return callbacks


class BatchGenerator(Sequence):
    def __init__(self, fnames, batch_size, shuffle, label=0, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.label = label
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.fnames) /self.batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        # Returns
            xs : (N, input_size, input_size, 3)
                rgb-ordered images
            ys : (N, input_size/4, input_size/4, 1)
        """
        batch_fnames = self.fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        xs = [cv2.imread(fname)[:,:,::-1] for fname in batch_fnames]
        xs = np.array([cv2.resize(img, (self.input_size,self.input_size)) for img in xs])
        ys = self.label + np.zeros((self.batch_size,
                                    int(self.input_size/4),
                                    int(self.input_size/4),
                                    1))
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)

