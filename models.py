# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import numpy as np
import os

from cartoon import PKG_ROOT
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

    # Block 1 : (256,256,3) -> (256,256,64)
    x = SpatialReflectionPadding(3)(x)
    x = Conv2D(64, (7, 7), strides=1, use_bias=True, padding='valid', name="conv1")(x)
    x = InstanceNormalization(name="in1")(x)
    x = Activation("relu")(x)

    # Block 2 : (256,256,64) -> (128,128,128)
    # Todo : strides (1->2)
    x = Conv2D(128, (3, 3), strides=1, use_bias=True, padding='same', name="conv2_1")(x)
    x = Conv2D(128, (3, 3), strides=1, use_bias=True, padding='same', name="conv2_2")(x)
    x = InstanceNormalization(name="in2")(x)
    x = Activation("relu")(x)

    # Block 3 : (128,128,128) -> (64,64,256)
    # Todo : strides (1->2)
    x = Conv2D(256, (3, 3), strides=1, use_bias=True, padding='same', name="conv3_1")(x)
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

    
    model = Model(img_input, x, name='cartoon_generator')
    # model.load_weights(h5_fname)
    return model


if __name__ == '__main__':
    model = cartoon_generator(input_size=256)
    model.summary()

    ys_torch = run_by_torch(load_net_in())
    print(ys_torch.shape)

    # 1st conv layer
    w1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "0.npy")), [2,3,1,0])
    b1 = np.load(os.path.join(PKG_ROOT, "Hayao", "1.npy"))
    model.get_layer(name="conv1").set_weights([w1, b1])

    # 1st in layer
    in1_a = np.load(os.path.join(PKG_ROOT, "Hayao", "2.npy"))
    in1_b = np.load(os.path.join(PKG_ROOT, "Hayao", "3.npy"))
    model.get_layer(name="in1").set_weights([in1_a, in1_b])
    
    w2_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "4.npy")), [2,3,1,0])
    b2_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "5.npy"))
    model.get_layer(name="conv2_1").set_weights([w2_1, b2_1])

    w2_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "6.npy")), [2,3,1,0])
    b2_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "7.npy"))
    model.get_layer(name="conv2_2").set_weights([w2_2, b2_2])

    # 2nd in layer
    in2_a = np.load(os.path.join(PKG_ROOT, "Hayao", "8.npy"))
    in2_b = np.load(os.path.join(PKG_ROOT, "Hayao", "9.npy"))
    model.get_layer(name="in2").set_weights([in2_a, in2_b])

    # Block3
    w3_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "10.npy")), [2,3,1,0])
    b3_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "11.npy"))
    model.get_layer(name="conv3_1").set_weights([w3_1, b3_1])
    w3_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "12.npy")), [2,3,1,0])
    b3_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "13.npy"))
    model.get_layer(name="conv3_2").set_weights([w3_2, b3_2])
    in3_a = np.load(os.path.join(PKG_ROOT, "Hayao", "14.npy"))
    in3_b = np.load(os.path.join(PKG_ROOT, "Hayao", "15.npy"))
    model.get_layer(name="in3").set_weights([in3_a, in3_b])

    # Block4
    w4_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "16.npy")), [2,3,1,0])
    b4_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "17.npy"))
    model.get_layer(name="conv4_1").set_weights([w4_1, b4_1])
    in4_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "18.npy"))
    in4_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "19.npy"))
    model.get_layer(name="in4_1").set_weights([in4_1a, in4_1b])

    w4_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "20.npy")), [2,3,1,0])
    b4_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "21.npy"))
    model.get_layer(name="conv4_2").set_weights([w4_2, b4_2])
    in4_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "22.npy"))
    in4_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "23.npy"))
    model.get_layer(name="in4_2").set_weights([in4_2a, in4_2b])

    # Block5
    w5_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "24.npy")), [2,3,1,0])
    b5_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "25.npy"))
    model.get_layer(name="conv5_1").set_weights([w5_1, b5_1])
    in5_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "26.npy"))
    in5_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "27.npy"))
    model.get_layer(name="in5_1").set_weights([in5_1a, in5_1b])

    w5_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "28.npy")), [2,3,1,0])
    b5_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "29.npy"))
    model.get_layer(name="conv5_2").set_weights([w5_2, b5_2])
    in5_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "30.npy"))
    in5_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "31.npy"))
    model.get_layer(name="in5_2").set_weights([in5_2a, in5_2b])

    # Block6
    w6_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "32.npy")), [2,3,1,0])
    b6_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "33.npy"))
    model.get_layer(name="conv6_1").set_weights([w6_1, b6_1])
    in6_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "34.npy"))
    in6_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "35.npy"))
    model.get_layer(name="in6_1").set_weights([in6_1a, in6_1b])
    w6_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "36.npy")), [2,3,1,0])
    b6_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "37.npy"))
    model.get_layer(name="conv6_2").set_weights([w6_2, b6_2])
    in6_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "38.npy"))
    in6_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "39.npy"))
    model.get_layer(name="in6_2").set_weights([in6_2a, in6_2b])

    # Block7
    w7_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "40.npy")), [2,3,1,0])
    b7_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "41.npy"))
    model.get_layer(name="conv7_1").set_weights([w7_1, b7_1])
    in7_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "42.npy"))
    in7_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "43.npy"))
    model.get_layer(name="in7_1").set_weights([in7_1a, in7_1b])
    w7_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "44.npy")), [2,3,1,0])
    b7_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "45.npy"))
    model.get_layer(name="conv7_2").set_weights([w7_2, b7_2])
    in7_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "46.npy"))
    in7_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "47.npy"))
    model.get_layer(name="in7_2").set_weights([in7_2a, in7_2b])

    # Block8
    w8_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "48.npy")), [2,3,1,0])
    b8_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "49.npy"))
    model.get_layer(name="conv8_1").set_weights([w8_1, b8_1])
    in8_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "50.npy"))
    in8_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "51.npy"))
    model.get_layer(name="in8_1").set_weights([in8_1a, in8_1b])
    w8_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "52.npy")), [2,3,1,0])
    b8_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "53.npy"))
    model.get_layer(name="conv8_2").set_weights([w8_2, b8_2])
    in8_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "54.npy"))
    in8_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "55.npy"))
    model.get_layer(name="in8_2").set_weights([in8_2a, in8_2b])

    # Block9
    w9_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "56.npy")), [2,3,1,0])
    b9_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "57.npy"))
    model.get_layer(name="conv9_1").set_weights([w9_1, b9_1])
    in9_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "58.npy"))
    in9_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "59.npy"))
    model.get_layer(name="in9_1").set_weights([in9_1a, in9_1b])
    w9_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "60.npy")), [2,3,1,0])
    b9_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "61.npy"))
    model.get_layer(name="conv9_2").set_weights([w9_2, b9_2])
    in9_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "62.npy"))
    in9_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "63.npy"))
    model.get_layer(name="in9_2").set_weights([in9_2a, in9_2b])

    # Block10
    w10_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "64.npy")), [2,3,1,0])
    b10_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "65.npy"))
    model.get_layer(name="conv10_1").set_weights([w10_1, b10_1])
    in10_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "66.npy"))
    in10_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "67.npy"))
    model.get_layer(name="in10_1").set_weights([in10_1a, in10_1b])
    w10_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "68.npy")), [2,3,1,0])
    b10_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "69.npy"))
    model.get_layer(name="conv10_2").set_weights([w10_2, b10_2])
    in10_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "70.npy"))
    in10_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "71.npy"))
    model.get_layer(name="in10_2").set_weights([in10_2a, in10_2b])

    # Block11
    w11_1 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "72.npy")), [2,3,1,0])
    b11_1 = np.load(os.path.join(PKG_ROOT, "Hayao", "73.npy"))
    model.get_layer(name="conv11_1").set_weights([w11_1, b11_1])
    in11_1a = np.load(os.path.join(PKG_ROOT, "Hayao", "74.npy"))
    in11_1b = np.load(os.path.join(PKG_ROOT, "Hayao", "75.npy"))
    model.get_layer(name="in11_1").set_weights([in11_1a, in11_1b])
    w11_2 = np.transpose(np.load(os.path.join(PKG_ROOT, "Hayao", "76.npy")), [2,3,1,0])
    b11_2 = np.load(os.path.join(PKG_ROOT, "Hayao", "77.npy"))
    model.get_layer(name="conv11_2").set_weights([w11_2, b11_2])
    in11_2a = np.load(os.path.join(PKG_ROOT, "Hayao", "78.npy"))
    in11_2b = np.load(os.path.join(PKG_ROOT, "Hayao", "79.npy"))
    model.get_layer(name="in11_2").set_weights([in11_2a, in11_2b])


    imgs = np.expand_dims(load_net_in(), axis=0)
    ys = model.predict(imgs)
    print(ys.shape)
    print(np.allclose(ys, ys_torch, rtol=1e-3, atol=1e-3))



