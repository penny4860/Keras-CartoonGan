# -*- coding: utf-8 -*-


import numpy as np
import glob
import tensorflow as tf
from cartoon.vgg import vgg_feat_extractor

np.random.seed(1337)

BATCH_SIZE = 8

def loss_func(y_true, y_pred):
    # 1. activate prediction & truth tensor
    # loss = tf.losses.mean_squared_error(y_true, y_pred) + 1e-4*tf.reduce_mean(tf.image.total_variation(y_pred))
    feat_model = vgg_feat_extractor()
    y_true_feat = feat_model(y_true)
    y_pred_feat = feat_model(y_pred)
    loss = tf.losses.absolute_difference(y_true_feat, y_pred_feat)
    return loss


if __name__ == '__main__':
    
    photo_fnames = glob.glob("../../dataset/cartoon_dataset/photo/*.*")
    from cartoon.models import cartoon_generator
    from cartoon.seq import IdenBatchGenerator, create_callbacks
    
    batch_gen = IdenBatchGenerator(photo_fnames, batch_size=BATCH_SIZE, shuffle=True, input_size=256)
    
    model_g = cartoon_generator()
    # model_g.load_weights("../params/Hayao.h5")
     
    model_g.compile(loss=loss_func,
                    optimizer=tf.keras.optimizers.Adam(1e-4))
    model_g.fit_generator(batch_gen,
                          steps_per_epoch=len(batch_gen),
                          callbacks=create_callbacks(saved_weights_name="init_generator.h5"),
                          validation_data = batch_gen,
                          validation_steps =len(batch_gen),
                          epochs=2000)
    
