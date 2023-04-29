import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import cv2
import os

# Define the paths for your data and output
data_path = r"C:\Users\junkinho\anomalib\datasets\MVTec\bottle\train\good"
output_path = r"C:\Users\junkinho\anomalib\input_anomalous1"

# Define the shape of your images
img_shape = (64, 64, 3)

# Define the generator model
def build_generator():
    model = Sequential()

    model.add(Dense(128 * 16 * 16, activation="relu", input_dim=100))
    model.add(Reshape((16, 16, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(100,))
    img = model(noise)

    return Model(noise, img)

# Define the discriminator model
def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise = Input(shape=(100,))
    img = generator(noise)
    validity = discriminator(img)
    return Model(noise, validity)

# Load the data
data = []
for filename in os.listdir(data_path):
    img = cv2.imread(os.path.join(data_path, filename))
    img = cv2.resize(img, (img_shape[0], img_shape[1]))
    data.append(img)
X_train = np.array(data)

# Normalize the images
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Set the hyperparameters for training
epochs = 10
batch_size = 32
save_interval = 1000
enlarged_size = (512,512)

# Build the GAN model
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Compile the GAN model
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the GAN model
for epoch in range(epochs):

    # Train the discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress
    print(f"Epoch {epoch}, Discriminator Loss: {d_loss[0]}, Discriminator Accuracy: {100 * d_loss[1]}, Generator Loss: {g_loss}")

    # Save the generated images 
    
    noise = np.random.normal(0, 1, (1, 100))
    gen_img = generator.predict(noise)[0]
    gen_img = (0.5 * gen_img + 0.5) * 255.0
    gen_img = gen_img.astype(np.uint8)
    gen_img=cv2.resize(gen_img, enlarged_size)
    output_file = os.path.join(output_path, f"generated_image_{epoch}.png")
    cv2.imwrite(output_file, gen_img)

