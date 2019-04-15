# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
from cartoon.utils import preprocess

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


class IdenBatchGenerator(Sequence):
    def __init__(self, fnames, batch_size, shuffle, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
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
        xs = np.array([preprocess(cv2.resize(img, (self.input_size,self.input_size))) for img in xs])
        return xs, xs

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)

# cartoon / edge-smoothing / photo
class CartoonBatchGenerator(Sequence):
    def __init__(self,
                 cartoon_fnames,
                 cartoon_smooth_fnames,
                 photo_fnames,
                 batch_size, input_size=256):
        self.cartoon_fnames = cartoon_fnames
        self.cartoon_smooth_fnames = cartoon_smooth_fnames
        self.photo_fnames = photo_fnames
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.on_epoch_end()

    def __len__(self):
        length = min(len(self.cartoon_fnames),
                     len(self.cartoon_smooth_fnames), 
                     len(self.photo_fnames))
        return int(length /self.batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        # Returns
            xs : (N, input_size, input_size, 3)
                rgb-ordered images
            ys : (N, input_size/4, input_size/4, 1)
        """
        def load(fnames):
            xs = [cv2.imread(fname)[:,:,::-1] for fname in fnames]
            xs = np.array([preprocess(cv2.resize(img, (self.input_size,self.input_size))) for img in xs])
            return xs
        batch_cartoon_fnames = self.cartoon_fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_cartoon_smooth_fnames = self.cartoon_smooth_fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_photo_fnames = self.photo_fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        return load(batch_cartoon_fnames), load(batch_cartoon_smooth_fnames), load(batch_photo_fnames)

    def on_epoch_end(self):
        np.random.shuffle(self.cartoon_fnames)
        np.random.shuffle(self.cartoon_smooth_fnames)
        np.random.shuffle(self.photo_fnames)



