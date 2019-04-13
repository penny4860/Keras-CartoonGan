# -*- coding: utf-8 -*-

import tensorflow as tf

tf.keras.applications.vgg19.preprocess_input

Model = tf.keras.models.Model

def vgg_feat_extractor():
    model = tf.keras.applications.vgg19.VGG19(weights='imagenet')
    model = Model(model.input,
                  model.get_layer('block4_conv4').output,
                  name='vgg_feat_extractor')
    model.trainable = False
    return model

def vgg_preprocess(xs):
    # xs : bgr-ordered, [0, 1]-ranged
    xs = xs * 255
    mean = [103.939, 116.779, 123.68]
    xs[..., 0] -= mean[0]
    xs[..., 1] -= mean[1]
    xs[..., 2] -= mean[2]
    return xs


if __name__ == '__main__':
    model = vgg_feat_extractor()
    # model.predict(np.zeros((1,224,224,3)))

    


