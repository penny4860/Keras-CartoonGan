# -*- coding: utf-8 -*-

from __future__ import print_function, division

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

#     target = tf.ones(y_shape)
#     d_loss = tf.keras.backend.binary_crossentropy(target, 
#                                                   y_pred)
#     loss = vgg_contents_loss + d_loss
    return vgg_contents_loss


class CartoonGan():
    def __init__(self):
        input_size = 256

        self.img_shape = (input_size, input_size, 3)
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)

        # Build and compile the discriminator
        self.discriminator = cartoon_discriminator(input_size)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = cartoon_generator(input_size)

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
        self.discriminator_generator = Model(p_tensor, validity)
        self.discriminator_generator.compile(loss=[g_loss_func, 'binary_crossentropy'],
                                             loss_weights=[1.0, 1.0],
                                             optimizer=optimizer)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    import glob
    photo_fnames = glob.glob("../../dataset/cartoon_dataset/photo/*.*")
    cartoon_fnames = glob.glob("../../dataset/cartoon_dataset/cartoon/*.*")
    cartoon_smooth_fnames = glob.glob("../../dataset/cartoon_dataset/cartoon_smooth/*.*")
    
    print(len(photo_fnames), len(cartoon_fnames), len(cartoon_smooth_fnames))

    gan = CartoonGan()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)

    from cartoon.seq import IdenBatchGenerator
    BATCH_SIZE = 8
    batch_gen = IdenBatchGenerator(photo_fnames,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   input_size=256)
    xs, _ = batch_gen[0]
    print(xs.shape)



