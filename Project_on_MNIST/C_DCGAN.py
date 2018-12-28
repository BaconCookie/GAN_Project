from __future__ import print_function, division

from time import time

from keras import callbacks
#from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

#########################################################################################
#                                                                                       #
# Combination of:                                                                       #
#                                                                                       #                                                                                    #
# Implementation of Deep Convolutional Generative Adversarial Network.                  #
# Code from: https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py    #
#   &                                                                                   #
# Implementation of Conditional Generative Adversarial Net.                             #
# Code from: https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py      #
#                                                                                       #
# combined by me                                                                        #
#########################################################################################

class C_DCGAN():

    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10  # add number of classes (labels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # # Add TensorBoard
        # file_path = "models/model_c_dcgan.hdf5"
        #
        # tensorboard = callbacks.TensorBoard(log_dir="logs/{}".format(time()), write_images=True)
        # # tensorboard = callbacks.TensorBoard(write_images=True)
        # checkpoint = callbacks.ModelCheckpoint(
        #     filepath=file_path,
        #     monitor='val_acc',
        #     save_best_only=True,
        #     save_weights_only=False,
        #     mode='max')
        #
        # callbacks = [tensorboard, checkpoint]


        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes as input:
        # - generated image (images_c_dcgan)
        # - the label of that image
        # to determine validity
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        generator = Sequential()

        # Generator consists of a deep convolutional net, as seen in the DCGAN example
        generator.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        generator.add(Reshape((7, 7, 128)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=3, padding="same"))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation("relu"))
        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=3, padding="same"))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Activation("relu"))
        generator.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        generator.add(Activation("tanh"))
        generator.add(Reshape(self.img_shape))

        generator.summary()

        noise = Input(shape=(self.latent_dim,))
        # add label
        label = Input(shape=(1,), dtype='int32')
        # add label_embedding
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        # calculate input for model
        model_input = multiply([noise, label_embedding])
        # generate an image
        img = generator(model_input)

        # The Model class adds training & evaluation routines to a Network.
        # A Network is a directed acyclic graph of layers.
        # It is the topological form of a "model". A Model is simply a Network with added training routines.

        return Model([noise, label], img)

    def build_discriminator(self):

        discriminator = Sequential()

        # Discriminator consists of a deep convolutional net, as seen in the DCGAN example
        discriminator.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
        # model.summary() # produces ValueError: This model has not yet been built. Build the model first!

        img = Input(shape=self.img_shape)
        # add label
        label = Input(shape=(1,), dtype='int32')
        # add label_embedding
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        # calculate input for model
        model_input = multiply([img, label_embedding])
        # calculate validity
        validity = discriminator(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images_c_dcgan
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images_c_dcgan
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator (real classified as ones and generated as zeros)
            # include labels
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Generate random labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            # The generator wants to be so good that the discriminator to mistakes its generated images for real
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                # save generator model
                file_path = "models/generator_c_dcgan.hdf5"
                self.generator.save(file_path)


    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images_c_dcgan 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images_c_dcgan/c_dc_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    c_dcgan = C_DCGAN()
    c_dcgan.train(epochs=51, batch_size=32, sample_interval=200)