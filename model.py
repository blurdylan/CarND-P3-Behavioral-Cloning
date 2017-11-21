
# coding: utf-8

# ## The `model.py` file
# This will be used to create the model which will be used along with keras, the main purpose of this file is to train the model using the data saved from our preprocessed file,
# Once the training is done the model and weights are saved in model.h5 file

# In[ ]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import cv2


# ## Preprocessing
# The preprocessing is done with these helper functions: change_brightness, resize_to_target_size and change_size_and_crop
# This will help change the images in a way that will best suite the model training.

# In[ ]:

rows, cols, ch = 64, 64, 3
TARGET_SIZE = (64, 64)

def resize_to_target_size(image):
    return cv2.resize(image, TARGET_SIZE)

def change_size_and_crop(image):
    # The input image of dimensions 160x320x3
    # Output image of size 64x64x3
    cropped_image = image[55:135, :, :]
    processed_image = resize_to_target_size(cropped_image)
    return processed_image

def change_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # The 0.25 used is to prevent the image to be completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def preprocess(image):
    # Preprocess image
    image = change_size_and_crop(image)
    image = image.astype(np.float32)

    # Normalize image
    image = image/255.0 - 0.5
    return image


'''
get the needed images from the dataset
and apply the preprocessing functions
on them
'''
def get_augmented_row(row):
    steering = row['steering']
    camera = np.random.choice(['center', 'left', 'right'])

    # change the steering angle relative to camera
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img("data/" + row[camera].strip())
    image = img_to_array(image)

    # Horizontally flip the image, helps to reduce left bias
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Apply image correctsions
    image = change_brightness(image)
    image = preprocess(image)
    return image, steering


# ## The model proper
# The model is initialized and trained with the parameters defined below

# In[4]:

def get_data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the data and generate augmented
        # images directly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def get_model():
    # initialize the model
    model = Sequential()

    # 1 shape is 32x32x32
    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # 2 shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # 3 shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # 5
    model.add(Dense(512))
    model.add(ELU())

    # Single output due to the regression
    model.add(Dense(1))

    # Adam optimizer
    model.compile(optimizer="adam", loss="mse")
    # Print out the model
    model.summary()

    return model

if __name__ == "__main__":
    BATCH_SIZE = 32

    # Get the data with labels from the csv file
    data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])
    
    # Dividing the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    # Memory free
    data_frame = None

    training_generator = get_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(validation_data, batch_size=BATCH_SIZE)

    model = get_model()

    # train 20000 samples in each epoch
    samples_per_epoch = (20000//BATCH_SIZE)*BATCH_SIZE

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)

    print("Saving model weights and configuration file...")

    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
        print("Done!")


