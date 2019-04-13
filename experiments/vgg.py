# -*- coding: utf-8 -*-

import tensorflow as tf

Model = tf.keras.models.Model

def vgg_feat_extractor():
    model = tf.keras.applications.vgg19.VGG19(weights='imagenet')
    model = Model(model.input,
                  model.get_layer('block4_conv4').output,
                  name='vgg_feat_extractor')
    model.summary()
    return model


if __name__ == '__main__':
    model = vgg_feat_extractor()

