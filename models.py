from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from collections import deque
import sys
import keras
from functools import partial

class ResearchModels:
    def __init__(self, nb_classes, model, seq_length, features_length=2048):

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        top3_acc = partial(keras.metrics.top_k_categorical_accuracy, k=3)

        top3_acc.__name__ = 'top3_acc'

        metrics = ['accuracy']
        if self.nb_classes >= 5:
            metrics.append(top3_acc)

        if model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            print(self.input_shape)
            self.model = self.lstm()

        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn()

        elif model == 'c3d':
            print("Loading C3D model")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.conv_3d()

        else:
            print("Unknown network.")
            sys.exit()

        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        print(self.model.summary())

    def lrcn(self):
        # CNN-LSTM network, also known as LRCN
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def lstm(self):
        # LSTM model
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False, input_shape=self.input_shape, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    # Build a 3D convolutional network
    def conv_3d(self):

        model = Sequential()
        model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (3, 3, 3), activation='relu'))
        model.add(Conv3D(128, (3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (2, 2, 2), activation='relu'))
        model.add(Conv3D(256, (2, 2, 2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
