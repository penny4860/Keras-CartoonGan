# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from cartoon.layers import SpatialReflectionPadding, InstanceNormalization
from cartoon.utils import load_net_in

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation
MaxPooling2D = tf.keras.layers.MaxPooling2D
Layer = tf.keras.layers.Layer
Model = tf.keras.models.Model
VGG19 = tf.keras.applications.vgg19.VGG19


def cartoon_generator(input_size=256):

    input_shape=[input_size,input_size,3]
    
    x = Input(shape=input_shape, name="input")
    img_input = x

    # Block 1 : (256,256,3) -> (256,256,64)
    x = SpatialReflectionPadding(3)(x)
    x = Conv2D(64, (7, 7), strides=1, use_bias=True, padding='valid', name="conv1")(x)
    x = InstanceNormalization(name="in1")(x)
    x = Activation("relu")(x)

    # Block 2 : (256,256,64) -> (128,128,128)
    x = Conv2D(128, (3, 3), strides=2, use_bias=True, padding='same', name="conv2_1")(x)
    x = Conv2D(128, (3, 3), strides=1, use_bias=True, padding='same', name="conv2_2")(x)
    x = InstanceNormalization(name="in2")(x)
    x = Activation("relu")(x)

    # Block 3 : (128,128,128) -> (64,64,256)
    x = Conv2D(256, (3, 3), strides=2, use_bias=True, padding='same', name="conv3_1")(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='same', name="conv3_2")(x)
    x = InstanceNormalization(name="in3")(x)
    x = Activation("relu")(x)
    res_in = x

    # Block 4 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv4_1")(x)
    x = InstanceNormalization(name="in4_1")(x)
    x = Activation("relu")(x)

    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv4_2")(x)
    x = InstanceNormalization(name="in4_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 5 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv5_1")(x)
    x = InstanceNormalization(name="in5_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv5_2")(x)
    x = InstanceNormalization(name="in5_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 6 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv6_1")(x)
    x = InstanceNormalization(name="in6_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv6_2")(x)
    x = InstanceNormalization(name="in6_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 7 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv7_1")(x)
    x = InstanceNormalization(name="in7_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv7_2")(x)
    x = InstanceNormalization(name="in7_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 8 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv8_1")(x)
    x = InstanceNormalization(name="in8_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv8_2")(x)
    x = InstanceNormalization(name="in8_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 9 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv9_1")(x)
    x = InstanceNormalization(name="in9_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv9_2")(x)
    x = InstanceNormalization(name="in9_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 10 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv10_1")(x)
    x = InstanceNormalization(name="in10_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv10_2")(x)
    x = InstanceNormalization(name="in10_2")(x)
    x = tf.keras.layers.Add()([x, res_in])
    res_in = x

    # Block 11 : (64,64,256) -> (64,64,256)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv11_1")(x)
    x = InstanceNormalization(name="in11_1")(x)
    x = Activation("relu")(x)
    x = SpatialReflectionPadding(1)(x)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='valid', name="conv11_2")(x)
    x = InstanceNormalization(name="in11_2")(x)
    x = tf.keras.layers.Add()([x, res_in])

    # Block 1 : (64,64,256) -> (128,128,128)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding="same", name="deconv1_1")(x)
    x = Conv2D(128, (3, 3), strides=1, use_bias=True, padding="same", name="deconv1_2")(x)
    x = InstanceNormalization(name="in_deconv1")(x)
    x = Activation("relu")(x)

    # Block 2 : (128,128,128) -> (256,256,64)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding="same", name="deconv2_1")(x)
    x = Conv2D(64, (3, 3), strides=1, use_bias=True, padding="same", name="deconv2_2")(x)
    x = InstanceNormalization(name="in_deconv2")(x)
    x = Activation("relu")(x)

    # Block 3 : (256,256,64) -> (256,256,3)
    x = SpatialReflectionPadding(3)(x)
    x = Conv2D(3, (7, 7), strides=1, use_bias=True, padding="valid", name="deconv3")(x)
    x = Activation("tanh")(x)
    
    model = Model(img_input, x, name='cartoon_generator')
    # model.load_weights(h5_fname)
    return model


def postprocess(ys):
    # bgr -> rgb
    ys = ys[...,::-1]
    # [0, 1]-range
    ys = ys * 0.5 + 0.5
    return ys


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    input_size = 512
    model_names = ["Hayao", "Hosoda", "Paprika", "Shinkai"]
    model = cartoon_generator(input_size=input_size)

    fig, ax = plt.subplots()
    plt.subplot(1, 5, 1)
    plt.axis('off')
    plt.title("input")
    plt.imshow(postprocess(load_net_in()))
    
    for i, model_name in enumerate(model_names):
        model.load_weights("params/{}.h5".format(model_name))
    
        imgs = np.expand_dims(load_net_in(desired_size=input_size), axis=0)
        ys = model.predict(imgs)
        y = postprocess(ys)[0]

        plt.subplot(1, 5, i+2)
        plt.axis('off')
        plt.title(model_name)
        plt.imshow(y)
    plt.show()

