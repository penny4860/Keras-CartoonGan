# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation
MaxPooling2D = tf.keras.layers.MaxPooling2D
Layer = tf.keras.layers.Layer
Model = tf.keras.models.Model
VGG19 = tf.keras.applications.vgg19.VGG19

from cartoon.models import cartoon_generator, cartoon_discriminator
from cartoon.vgg import vgg_feat_extractor


def g_loss_func(y_true, y_pred):
    # 1. activate prediction & truth tensor
    # loss = tf.losses.mean_squared_error(y_true, y_pred) + 1e-4*tf.reduce_mean(tf.image.total_variation(y_pred))
    feat_model = vgg_feat_extractor()
    y_true_feat = feat_model(y_true)
    y_pred_feat = feat_model(y_pred)
    vgg_contents_loss = tf.losses.absolute_difference(y_true_feat, y_pred_feat)
    return vgg_contents_loss


class CartoonGan():
    def __init__(self, pretrained_generator_fname="../../dataset/cartoon_dataset/params/init_generator.h5"):
        input_size = 256

        self.img_shape = (input_size, input_size, 3)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)

        # Build and compile the discriminator
        self.discriminator = cartoon_discriminator(input_size)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer)

        # Build the generator
        self.generator = cartoon_generator(input_size)
        self.generator.load_weights(pretrained_generator_fname)

        # The generator takes noise as input and generates imgs
        p_tensor = Input(shape=self.img_shape)
        generated_catroon_tensor = self.generator(p_tensor)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_catroon_tensor)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # D(G(x))
        self.discriminator_generator = Model(p_tensor,
                                             outputs=[generated_catroon_tensor, validity])
        self.discriminator_generator.compile(loss=[g_loss_func, 'binary_crossentropy'],
                                             loss_weights=[10.0, 1.0],
                                             optimizer=optimizer)

    def train(self, batch_generator, epochs=50000):
        # Adversarial loss ground truths
        valid = np.ones((batch_generator.batch_size,) + (64,64,1))
        fake = np.zeros((batch_generator.batch_size,) + (64,64,1))

        for batch_i, (cartoon_imgs, cartoon_smooth_imgs, photo_imgs) in enumerate(batch_generator):
        
            # ----------------------
            #  Train Discriminators
            # ----------------------
            gen_cartoon_imgs = self.generator.predict(photo_imgs)
        
            d_loss_real = self.discriminator.train_on_batch(cartoon_imgs, valid)
            d_loss_smooth = self.discriminator.train_on_batch(cartoon_smooth_imgs, fake)
            d_loss_fake = self.discriminator.train_on_batch(gen_cartoon_imgs, fake)
            d_loss = (d_loss_real + d_loss_smooth + d_loss_fake) / 3
        
            # ------------------
            #  Train Generators
            # ------------------
            g_loss = self.discriminator_generator.train_on_batch(photo_imgs,
                                                                 [photo_imgs, valid])
            print("{}-th step, d_loss: {}, g_loss: {}".format(batch_i, d_loss, g_loss))

            if batch_i % 10 == 0:
                self.sample_images(batch_i)
                self.generator.save_weights("{}.h5".format(batch_i))
            
            if batch_i == epochs:
                break
                
                

    def sample_images(self, epoch):
        from cartoon.utils import preprocess, postprocess
        os.makedirs('generated_imgs', exist_ok=True)
        
        fname = "../../dataset/cartoon_dataset/photo/raccoon-1.jpg"
        img = cv2.resize(cv2.imread(fname)[:,:,::-1], (256,256))
        imgs = preprocess(np.expand_dims(img, axis=0))
        gen_imgs = self.generator.predict(imgs)
        gen_img = postprocess(gen_imgs)[0]

        fig, ax = plt.subplots()
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title("input photo")
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title("cartoonized")
        plt.imshow(gen_img)

        fig.savefig("generated_imgs/{}.jpg".format(epoch))
        plt.close()



if __name__ == '__main__':
    import glob
    photo_fnames = glob.glob("../../dataset/cartoon_dataset/photo/*.*")
    cartoon_fnames = glob.glob("../../dataset/cartoon_dataset/cartoon/*.*")
    cartoon_smooth_fnames = glob.glob("../../dataset/cartoon_dataset/cartoon_smooth/*.*")
    
    from cartoon.seq import CartoonBatchGenerator
    batch_generator = CartoonBatchGenerator(cartoon_fnames, cartoon_smooth_fnames, photo_fnames, batch_size=8)
    gan = CartoonGan()
    gan.train(batch_generator)




