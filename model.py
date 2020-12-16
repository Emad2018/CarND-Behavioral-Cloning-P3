import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from math import ceil
from random import shuffle
import keras
import tensorflow


 
print(keras.__version__)
print(tensorflow.__version__)
data_dir="./data/"
samples = []
start=1
with open(data_dir+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(start):
            start=0
            continue
        samples.append(line)
start=1
with open(data_dir+'driving_log1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(start):
            start=0
            continue
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


print(len(samples))
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_img_dir = data_dir+'IMG/'+batch_sample[0].split('/')[-1]
                left_img_dir  = data_dir+'IMG/'+batch_sample[1].split('/')[-1]
                right_img_dir  = data_dir+'IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_img_dir)
                left_image = cv2.imread(left_img_dir)
                right_image = cv2.imread(right_img_dir)
                steering_center = float(batch_sample[3])
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                images.extend([center_image,left_image,right_image])
                angles.extend([steering_center,steering_left,steering_right])
                
            augmented_images,augmented_angles=[],[]
            for image,angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            images=[]
            angles=[]   
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, 5, 5, subsample=( 2, 2),activation="relu"))
model.add(Conv2D(36, 5, 5, subsample=( 2, 2),activation="relu"))
model.add(Conv2D(48, 5, 5, subsample=( 2, 2),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, 3,activation="relu" ))
model.add(Conv2D(64, 3, 3,activation="relu" ))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)
model.save("model.h5")
print("end")