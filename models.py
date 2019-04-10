

import tensorflow as tf
import keras
from cartoon import USE_TF_KERAS
from cartoon.layers import SpatialReflectionPadding, InstanceNormalization

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
    x = Conv2D(64, (7, 7), strides=1, use_bias=True, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation("relu")(x)
    # (112,112,32)

    model = Model(img_input, x, name='cartoon_generator')
    # model.load_weights(h5_fname)
    return model


if __name__ == '__main__':
    model = cartoon_generator(input_size=256)
    model.summary()
