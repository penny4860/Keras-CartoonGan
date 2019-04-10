# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import numpy as np

from cartoon import USE_TF_KERAS
from cartoon.layers import SpatialReflectionPadding, InstanceNormalization
from cartoon.utils import load_net_in
from cartoon.utils import run_by_torch


if USE_TF_KERAS:
    Input = tf.keras.layers.Input
    Conv2D = tf.keras.layers.Conv2D
    DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Activation = tf.keras.layers.Activation
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    Layer = tf.keras.layers.Layer
    Model = tf.keras.models.Model
    VGG19 = tf.keras.applications.vgg19.VGG19
else:
    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    DepthwiseConv2D = keras.layers.DepthwiseConv2D
    BatchNormalization = keras.layers.BatchNormalization
    Activation = keras.layers.Activation
    MaxPooling2D = keras.layers.MaxPooling2D
    Layer = keras.layers.Layer
    Model = keras.models.Model
    VGG19 = keras.applications.vgg19.VGG19

def cartoon_generator(input_size=256):

    input_shape=[input_size,input_size,3]
    
    x = Input(shape=input_shape, name="input")
    img_input = x

    # Block 1
    x = SpatialReflectionPadding(3)(x)
    x = Conv2D(64, (7, 7), strides=1, use_bias=True, padding='valid', name="conv1")(x)
#     x = InstanceNormalization()(x)
#     x = Activation("relu")(x)

    model = Model(img_input, x, name='cartoon_generator')
    # model.load_weights(h5_fname)
    return model


if __name__ == '__main__':
    model = cartoon_generator(input_size=256)
    model.summary()

    ys_torch = run_by_torch(load_net_in())
    print(ys_torch.shape)

    import os
    from cartoon import PKG_ROOT
    w = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "0.npy")), [2,3,1,0])
    b = np.load(os.path.join(PKG_ROOT, "Hayao", "1.npy"))
    model.get_layer(name="conv1").set_weights([w, b])

    imgs = np.expand_dims(load_net_in(), axis=0)
    ys = model.predict(imgs)
    print(ys.shape)
    print(np.allclose(ys, ys_torch, rtol=1e-3, atol=1e-3))



