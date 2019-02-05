from __future__ import print_function, division
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding image for that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        print("build_generator, label_embedding", label_embedding)

        model_input = multiply([noise, label_embedding])
        print("build_generator, model_input", model_input)

        img = model(model_input)
        print("build_generator, img", img)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        # print("self.img_shape", self.img_shape)
        # print("np.prod(self.img_shape)", np.prod(self.img_shape))

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])
        # print("model_input", model_input)
        validity = model(model_input)
        # print("validity", validity)
        # print("Model([img, label], validity)", Model([img, label], validity))
        return Model([img, label], validity)

    def get_label(self, genre):
        if genre == 'black':
            label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif genre == 'core':
            label = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif genre == 'death':
            label = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif genre == 'doom':
            label = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif genre == 'gothic':
            label = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif genre == 'heavy':
            label = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif genre == 'pagan':
            label = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif genre == 'power':
            label = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif genre == 'progressive':
            label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif genre == 'thrash':
            label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        else:
            return KeyError
        return label

    def load_data(self, n_images):
        n_images = n_images - 1
        data = np.empty((n_images, 64, 128, 3), dtype='float32')  # number of images, n channels (3 = RGB), w, h
        label = np.empty((n_images, 10), dtype='uint8')
        i = 0
        for filename in glob.glob('./preprocessed_imgs_all/*.jpg'):
            if i == n_images:
                break
            try:
                img = Image.open(filename)
                arr = np.asarray(img, dtype='float32')
                genre = filename.rsplit('/', 1)[-1].rsplit('_', 1)[0]
                # band_nr = filename.rsplit('/', 1)[-1].rsplit('_', 1)[1].rsplit('.', 1)[0]
                # print(arr.shape)
                data[i, :, :, :] = arr
                label[i] = self.get_label(genre)
                i += 1
            except OSError:
                print('OSError caused by: ', img)
            except KeyError:
                print('OSError caused by: ', img, genre)
        return data, label

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train) = self.load_data(43511)
        # print(X_train.shape)
        # print(y_train.shape)

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)  # horizontal to vertical vector

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images (instead of shuffling the data)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            # print('IDX ',idx)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # # Rescale images_cgan 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].set_title("class: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images_cgan_BaLo/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    BaLo = CGAN()

    BaLo.train(epochs=1, batch_size=32, sample_interval=50)  # TODO adjust batch size to what fits in memory
