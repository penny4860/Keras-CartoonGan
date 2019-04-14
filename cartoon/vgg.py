# -*- coding: utf-8 -*-

import tensorflow as tf
from cartoon.layers import VggPreprocess

tf.keras.applications.vgg19.preprocess_input
Model = tf.keras.models.Model

# Todo : preprocess layer를 parameter로 선택
def vgg_feat_extractor(input_size=256):
    def _build_model(input_shape):
        img_input = tf.keras.layers.Input(shape=input_shape, name="input")
        
        x = VggPreprocess()(img_input)
        base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv4').output)
        x = model(x)

        feat_model = Model(img_input, x, name='vgg_feat_extractor')
        return feat_model
    input_shape=[input_size,input_size,3]
    model = _build_model(input_shape)
    model.trainable = False
    return model


if __name__ == '__main__':
    model = vgg_feat_extractor()
    model.summary()
    # model.predict(np.zeros((1,224,224,3)))

    


