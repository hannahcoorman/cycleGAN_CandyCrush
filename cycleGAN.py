import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE

# Load images from folder

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Load the images

summer = "pathname"
winter = "pathname"

images_summer = load_images_from_folder(summer)
images_winter = load_images_from_folder(winter)

""" Data augmentation of the images:
    - Horizontal shift
    - Vertical shift
    - Horizontal flip
    - Zoom
    - Brightness
"""
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def datagenerator(data, augmented, nr):
    for i in data:
        image = i.copy()
        image = np.array(image, dtype=np.uint8) 
        samples = expand_dims(image, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-1,1],height_shift_range=0.01,
                                  horizontal_flip=True,  
                                  zoom_range=[0.97,1.03], brightness_range=[0.9,1.1])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        nr_plot = 0
        for i in range(nr):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # append data
            augmented.append(image)
    return augmented

def augment_data(data, nr):
    augmented = []
    # Data augmentation for Summer
    augmented = datagenerator(data, augmented, nr)
    X = np.array(augmented)
    return X

images_summer = augment_data(images_summer, 128)		# Choose the number of augmented images 
images_winter = augment_data(images_winter, 128)		# (if you have more data, the number does not need to be this high)


# Resize the images to equal size

def Resize_and_crop(images):
  RESIZE_WIDTH = 300
  RESIZE_HEIGHT = 300

  imgs = []
  x1 = 0
  x2 = 0
  y1 = 0
  y2 = 0
  for im in images:
    width = int(im.shape[1])
    height = int(im.shape[0])    
    # Setting the points for cropped image
    if width > height:
      x1 = int((width - height)/2)      
      y1 = height
      x2 = int((width - height)/2 + height)
      y2 = 0
      crop_im = im[y2:y1, x1:x2]
      resized = cv2.resize(crop_im, (RESIZE_HEIGHT,RESIZE_WIDTH), interpolation = cv2.INTER_AREA)
      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
      imgs.append(resized)
    if width < height:
      x1 = 0
      y1 = int((height - width)/2 + width)
      x2 = width
      y2 = int((height - width)/2)
      crop_im = im[y2:y1, x1:x2]
      resized = cv2.resize(crop_im, (RESIZE_HEIGHT,RESIZE_WIDTH), interpolation = cv2.INTER_AREA)
      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
      imgs.append(resized)
    if width == height: 
      resized = cv2.resize(im, (RESIZE_HEIGHT,RESIZE_WIDTH), interpolation = cv2.INTER_AREA)
      resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
      imgs.append(resized)
    
  return imgs

# Resize the images

S = []
W = []
S = Resize_and_crop(images_summer)
S = np.array(S)
W = Resize_and_crop(images_winter)
W = np.array(W)

S, s = S, np.ones((len(S),), dtype=int)
W, w = W, np.zeros((len(W),), dtype=int)


######################################
############INPUT PIPELINE############
######################################

train_S, test_S, train_s, test_s = train_test_split(S, s, test_size=0.2, random_state=42)
train_W, test_W, train_w, test_w = train_test_split(W, w, test_size=0.2, random_state=42)

train_summer = tf.data.Dataset.from_tensor_slices((train_S, train_s))
train_winter = tf.data.Dataset.from_tensor_slices((train_W, train_w))
test_summer = tf.data.Dataset.from_tensor_slices((test_S, test_s))
test_winter = tf.data.Dataset.from_tensor_slices((test_W, test_w))

BUFFER_SIZE = 1000
BATCH_SIZE = 1
ORIGINAL = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
   cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
   return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)
  # random mirroring
  image = tf.image.random_flip_left_right(image)
  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image


train_summer = train_summer.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

train_winter = train_winter.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_summer = test_summer.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_winter = test_winter.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

sample_summer = next(iter(train_summer))
sample_winter = next(iter(train_winter))

sample_summer = tf.image.resize(sample_summer, (256,256))
sample_winter = tf.image.resize(sample_winter, (256,256))

###############################################
########IMPORT AND REUSE PIX2PIX MODELS########
###############################################

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

###############################################
################LOSS FUNCTIONS#################
###############################################

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 40

def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  return prediction[0]


@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


############################################
##############TRAIN MODEL###################
############################################

for epoch in range(EPOCHS):
  	start = time.time()

  	n = 0
  	for image_x, image_y in tf.data.Dataset.zip((train_summer, train_winter)):
    	train_step(image_x, image_y)
    	if n % 10 == 0:
      		print ('.', end='')
    	n += 1

  	clear_output(wait=True)
  	# Using a consistent image (sample_summer) so that the progress of the model is clearly visible.
  	generate_images(generator_g, sample_summer)

  	if (epoch + 1) % 5 == 0:
    	ckpt_save_path = ckpt_manager.save()
    	print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  	print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))



#######################################
##############RUN MODEL################
#######################################

# Run the trained model on the test dataset:
for inp in test_summer.take(5):
    inp = tf.image.resize(inp, (256,256))
    generate_images(generator_g, inp)

# Run the trained model on the test dataset
for inp in test_winter.take(5):
  	inp = tf.image.resize(inp, (256,256))
  	generate_images(generator_f, inp)