'''
Created on 8 Mar 2021

@author: rapha
'''
'''
Created on 26 Feb 2021

@author: rapha
'''

import os, signal
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow import keras

train_dir = 'MMA/train/'
test_dir = 'MMA/test/'
valid_dir = 'MMA/valid/'


def trainML(newTrain=True):
    # All images will be rescalled by 1./255

    batch = 128
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range = 20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    test_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range = 20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(300, 300),
                                                        color_mode='rgb',
                                                        batch_size = batch,
                                                        class_mode='categorical',
                                                        shuffle=True)
    
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                    target_size=(48, 48),
                                    color_mode='rgb',
                                    batch_size = batch,
                                    class_mode='categorical',
                                    shuffle=True)
    
    test_generator = test_datagen.flow_from_directory(train_dir,
                                                        target_size=(300, 300),
                                                        color_mode='rgb',
                                                        batch_size = batch,
                                                        class_mode='categorical',
                                                        shuffle=True)

    
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=0.0001,
            patience=10, verbose=1, mode='auto',
            baseline=None, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            verbose=1,
            mode='auto')
    ]

    if newTrain is True:
        model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(64, (3,3),activation='relu',input_shape=(300, 300, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3,3),activation='relu',input_shape=(300, 300, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3,3),activation='relu',input_shape=(300, 300, 3)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.BatchNormalization(),
            
            # The second convolution
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),
            
            # The third convolution
            tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),
            
            # The third convolution
            tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),
            
            # The fourth convolution
            tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.BatchNormalization(),
            
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            
            # 512 neuron hidden layer
            #tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
    
        model.summary()
        model.compile(loss = 'categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])
        
        print('Starting New Training')
        history = model.fit(train_generator,
                            epochs=3,
                            validation_data = valid_generator,
                            shuffle=True,
                            batch_size=batch,
                            steps_per_epoch=92976//batch,
                            validation_steps=17364//batch,
                            callbacks=callbacks_list,
                            verbose = 1)

        model.save('Emotion Classifier.h5')
        model.summary()
        print('Evaluating new training:')
        nb_samples = len(valid_generator)
        model.evaluate(valid_generator, steps=nb_samples)
    else: # newTrain is False
        model = tf.keras.models.load_model('Emotion Classifier.h5')
        model.summary()
        print('Resume Training')
#         history = model.fit(train_generator,
#                             epochs=3,
#                             validation_data = valid_generator,
#                             shuffle=True,
#                             batch_size=batch,
#                             steps_per_epoch=92976//batch,
#                             validation_steps=17364//batch,
#                             callbacks=callbacks_list,
#                             verbose = 1)


        #model.save('Emotion Classifier.h5')
        #print('Model saved')
        model.summary()
        print('Evaluating continued training:')
        nb_samples = len(valid_generator)
        model.evaluate(valid_generator, steps=nb_samples)
        print('ready')
    return model

def predictML(model):

    print()
    print('Fetching Pictures to be Predicted')
    list_image_tensor = []
    for r, dir, file in os.walk(test_dir):
        for f in file:
            img = image.load_img(test_dir + f, target_size=(48, 48))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            image_tensor = np.vstack([x])
            list_image_tensor.append(image_tensor)

    #print('')
    #print(list_image_tensor)
    print()
    print('Creating prediction')
    list_prediction = []
    for img in list_image_tensor:
        print('Predicting image... ')
        classes = model.predict(img)
        #print(classes)
        list_prediction.append(classes[0])
    x = pd.DataFrame(list_prediction, columns=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])

    x = x.round(decimals=0)
    x.to_csv('predict_img.csv', sep=';', decimal=',')
    print(x)
    print('Ready')

    return model

m = trainML(newTrain=True)
#predictML(m)

os.kill(os.getpid(), signal.SIGKILL)