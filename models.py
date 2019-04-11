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


def load_params(model, params_root):
    
    # 1st conv layer
    w1 = np.transpose(np.load(os.path.join(params_root, "0.npy")), [2,3,1,0])
    b1 = np.load(os.path.join(params_root, "1.npy"))
    model.get_layer(name="conv1").set_weights([w1, b1])

    # 1st in layer
    in1_a = np.load(os.path.join(params_root, "2.npy"))
    in1_b = np.load(os.path.join(params_root, "3.npy"))
    model.get_layer(name="in1").set_weights([in1_a, in1_b])
    
    w2_1 = np.transpose(np.load(os.path.join(params_root, "4.npy")), [2,3,1,0])
    b2_1 = np.load(os.path.join(params_root, "5.npy"))
    model.get_layer(name="conv2_1").set_weights([w2_1, b2_1])

    w2_2 = np.transpose(np.load(os.path.join(params_root, "6.npy")), [2,3,1,0])
    b2_2 = np.load(os.path.join(params_root, "7.npy"))
    model.get_layer(name="conv2_2").set_weights([w2_2, b2_2])

    # 2nd in layer
    in2_a = np.load(os.path.join(params_root, "8.npy"))
    in2_b = np.load(os.path.join(params_root, "9.npy"))
    model.get_layer(name="in2").set_weights([in2_a, in2_b])

    # Block3
    w3_1 = np.transpose(np.load(os.path.join(params_root, "10.npy")), [2,3,1,0])
    b3_1 = np.load(os.path.join(params_root, "11.npy"))
    model.get_layer(name="conv3_1").set_weights([w3_1, b3_1])
    w3_2 = np.transpose(np.load(os.path.join(params_root, "12.npy")), [2,3,1,0])
    b3_2 = np.load(os.path.join(params_root, "13.npy"))
    model.get_layer(name="conv3_2").set_weights([w3_2, b3_2])
    in3_a = np.load(os.path.join(params_root, "14.npy"))
    in3_b = np.load(os.path.join(params_root, "15.npy"))
    model.get_layer(name="in3").set_weights([in3_a, in3_b])

    # Block4
    w4_1 = np.transpose(np.load(os.path.join(params_root, "16.npy")), [2,3,1,0])
    b4_1 = np.load(os.path.join(params_root, "17.npy"))
    model.get_layer(name="conv4_1").set_weights([w4_1, b4_1])
    in4_1a = np.load(os.path.join(params_root, "18.npy"))
    in4_1b = np.load(os.path.join(params_root, "19.npy"))
    model.get_layer(name="in4_1").set_weights([in4_1a, in4_1b])

    w4_2 = np.transpose(np.load(os.path.join(params_root, "20.npy")), [2,3,1,0])
    b4_2 = np.load(os.path.join(params_root, "21.npy"))
    model.get_layer(name="conv4_2").set_weights([w4_2, b4_2])
    in4_2a = np.load(os.path.join(params_root, "22.npy"))
    in4_2b = np.load(os.path.join(params_root, "23.npy"))
    model.get_layer(name="in4_2").set_weights([in4_2a, in4_2b])

    # Block5
    w5_1 = np.transpose(np.load(os.path.join(params_root, "24.npy")), [2,3,1,0])
    b5_1 = np.load(os.path.join(params_root, "25.npy"))
    model.get_layer(name="conv5_1").set_weights([w5_1, b5_1])
    in5_1a = np.load(os.path.join(params_root, "26.npy"))
    in5_1b = np.load(os.path.join(params_root, "27.npy"))
    model.get_layer(name="in5_1").set_weights([in5_1a, in5_1b])

    w5_2 = np.transpose(np.load(os.path.join(params_root, "28.npy")), [2,3,1,0])
    b5_2 = np.load(os.path.join(params_root, "29.npy"))
    model.get_layer(name="conv5_2").set_weights([w5_2, b5_2])
    in5_2a = np.load(os.path.join(params_root, "30.npy"))
    in5_2b = np.load(os.path.join(params_root, "31.npy"))
    model.get_layer(name="in5_2").set_weights([in5_2a, in5_2b])

    # Block6
    w6_1 = np.transpose(np.load(os.path.join(params_root, "32.npy")), [2,3,1,0])
    b6_1 = np.load(os.path.join(params_root, "33.npy"))
    model.get_layer(name="conv6_1").set_weights([w6_1, b6_1])
    in6_1a = np.load(os.path.join(params_root, "34.npy"))
    in6_1b = np.load(os.path.join(params_root, "35.npy"))
    model.get_layer(name="in6_1").set_weights([in6_1a, in6_1b])
    w6_2 = np.transpose(np.load(os.path.join(params_root, "36.npy")), [2,3,1,0])
    b6_2 = np.load(os.path.join(params_root, "37.npy"))
    model.get_layer(name="conv6_2").set_weights([w6_2, b6_2])
    in6_2a = np.load(os.path.join(params_root, "38.npy"))
    in6_2b = np.load(os.path.join(params_root, "39.npy"))
    model.get_layer(name="in6_2").set_weights([in6_2a, in6_2b])

    # Block7
    w7_1 = np.transpose(np.load(os.path.join(params_root, "40.npy")), [2,3,1,0])
    b7_1 = np.load(os.path.join(params_root, "41.npy"))
    model.get_layer(name="conv7_1").set_weights([w7_1, b7_1])
    in7_1a = np.load(os.path.join(params_root, "42.npy"))
    in7_1b = np.load(os.path.join(params_root, "43.npy"))
    model.get_layer(name="in7_1").set_weights([in7_1a, in7_1b])
    w7_2 = np.transpose(np.load(os.path.join(params_root, "44.npy")), [2,3,1,0])
    b7_2 = np.load(os.path.join(params_root, "45.npy"))
    model.get_layer(name="conv7_2").set_weights([w7_2, b7_2])
    in7_2a = np.load(os.path.join(params_root, "46.npy"))
    in7_2b = np.load(os.path.join(params_root, "47.npy"))
    model.get_layer(name="in7_2").set_weights([in7_2a, in7_2b])

    # Block8
    w8_1 = np.transpose(np.load(os.path.join(params_root, "48.npy")), [2,3,1,0])
    b8_1 = np.load(os.path.join(params_root, "49.npy"))
    model.get_layer(name="conv8_1").set_weights([w8_1, b8_1])
    in8_1a = np.load(os.path.join(params_root, "50.npy"))
    in8_1b = np.load(os.path.join(params_root, "51.npy"))
    model.get_layer(name="in8_1").set_weights([in8_1a, in8_1b])
    w8_2 = np.transpose(np.load(os.path.join(params_root, "52.npy")), [2,3,1,0])
    b8_2 = np.load(os.path.join(params_root, "53.npy"))
    model.get_layer(name="conv8_2").set_weights([w8_2, b8_2])
    in8_2a = np.load(os.path.join(params_root, "54.npy"))
    in8_2b = np.load(os.path.join(params_root, "55.npy"))
    model.get_layer(name="in8_2").set_weights([in8_2a, in8_2b])

    # Block9
    w9_1 = np.transpose(np.load(os.path.join(params_root, "56.npy")), [2,3,1,0])
    b9_1 = np.load(os.path.join(params_root, "57.npy"))
    model.get_layer(name="conv9_1").set_weights([w9_1, b9_1])
    in9_1a = np.load(os.path.join(params_root, "58.npy"))
    in9_1b = np.load(os.path.join(params_root, "59.npy"))
    model.get_layer(name="in9_1").set_weights([in9_1a, in9_1b])
    w9_2 = np.transpose(np.load(os.path.join(params_root, "60.npy")), [2,3,1,0])
    b9_2 = np.load(os.path.join(params_root, "61.npy"))
    model.get_layer(name="conv9_2").set_weights([w9_2, b9_2])
    in9_2a = np.load(os.path.join(params_root, "62.npy"))
    in9_2b = np.load(os.path.join(params_root, "63.npy"))
    model.get_layer(name="in9_2").set_weights([in9_2a, in9_2b])

    # Block10
    w10_1 = np.transpose(np.load(os.path.join(params_root, "64.npy")), [2,3,1,0])
    b10_1 = np.load(os.path.join(params_root, "65.npy"))
    model.get_layer(name="conv10_1").set_weights([w10_1, b10_1])
    in10_1a = np.load(os.path.join(params_root, "66.npy"))
    in10_1b = np.load(os.path.join(params_root, "67.npy"))
    model.get_layer(name="in10_1").set_weights([in10_1a, in10_1b])
    w10_2 = np.transpose(np.load(os.path.join(params_root, "68.npy")), [2,3,1,0])
    b10_2 = np.load(os.path.join(params_root, "69.npy"))
    model.get_layer(name="conv10_2").set_weights([w10_2, b10_2])
    in10_2a = np.load(os.path.join(params_root, "70.npy"))
    in10_2b = np.load(os.path.join(params_root, "71.npy"))
    model.get_layer(name="in10_2").set_weights([in10_2a, in10_2b])

    # Block11
    w11_1 = np.transpose(np.load(os.path.join(params_root, "72.npy")), [2,3,1,0])
    b11_1 = np.load(os.path.join(params_root, "73.npy"))
    model.get_layer(name="conv11_1").set_weights([w11_1, b11_1])
    in11_1a = np.load(os.path.join(params_root, "74.npy"))
    in11_1b = np.load(os.path.join(params_root, "75.npy"))
    model.get_layer(name="in11_1").set_weights([in11_1a, in11_1b])
    w11_2 = np.transpose(np.load(os.path.join(params_root, "76.npy")), [2,3,1,0])
    b11_2 = np.load(os.path.join(params_root, "77.npy"))
    model.get_layer(name="conv11_2").set_weights([w11_2, b11_2])
    in11_2a = np.load(os.path.join(params_root, "78.npy"))
    in11_2b = np.load(os.path.join(params_root, "79.npy"))
    model.get_layer(name="in11_2").set_weights([in11_2a, in11_2b])

    # Deconv Block1
    w_d3_1 = np.transpose(np.load(os.path.join(params_root, "80.npy")), [2,3,1,0])
    b_d3_1 = np.load(os.path.join(params_root, "81.npy"))
    model.get_layer(name="deconv1_1").set_weights([w_d3_1, b_d3_1])
    w_d3_2 = np.transpose(np.load(os.path.join(params_root, "82.npy")), [2,3,1,0])
    b_d3_2 = np.load(os.path.join(params_root, "83.npy"))
    model.get_layer(name="deconv1_2").set_weights([w_d3_2, b_d3_2])
    in_d3_a = np.load(os.path.join(params_root, "84.npy"))
    in_d3_b = np.load(os.path.join(params_root, "85.npy"))
    model.get_layer(name="in_deconv1").set_weights([in_d3_a, in_d3_b])

    # Deconv Block2
    w_d3_1 = np.transpose(np.load(os.path.join(params_root, "86.npy")), [2,3,1,0])
    b_d3_1 = np.load(os.path.join(params_root, "87.npy"))
    model.get_layer(name="deconv2_1").set_weights([w_d3_1, b_d3_1])
    w_d3_2 = np.transpose(np.load(os.path.join(params_root, "88.npy")), [2,3,1,0])
    b_d3_2 = np.load(os.path.join(params_root, "89.npy"))
    model.get_layer(name="deconv2_2").set_weights([w_d3_2, b_d3_2])
    in_d3_a = np.load(os.path.join(params_root, "90.npy"))
    in_d3_b = np.load(os.path.join(params_root, "91.npy"))
    model.get_layer(name="in_deconv2").set_weights([in_d3_a, in_d3_b])

    w_d3_1 = np.transpose(np.load(os.path.join(params_root, "92.npy")), [2,3,1,0])
    b_d3_1 = np.load(os.path.join(params_root, "93.npy"))
    model.get_layer(name="deconv3").set_weights([w_d3_1, b_d3_1])
    return model

if __name__ == '__main__':
    model = cartoon_generator(input_size=256)
    model = load_params(model, params_root=os.path.join(PKG_ROOT, "Hayao"))

    ys_torch = run_by_torch(load_net_in())
    print(ys_torch.shape)
    
    imgs = np.expand_dims(load_net_in(), axis=0)
    ys = model.predict(imgs)

    ys = postprocess(ys)
    ys_torch = postprocess(ys_torch)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("input")
    plt.imshow(postprocess(load_net_in()))
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title("keras output")
    plt.imshow(ys[0])
    plt.subplot(1, 3, 3)
    plt.axis('off')    
    plt.title("pytorch output")
    plt.imshow(ys_torch[0])
    plt.show()

