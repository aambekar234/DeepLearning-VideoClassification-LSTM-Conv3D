import os
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm
import pandas as pd
import random

class DataLoader:

    def __init__(self, data_path, data_type, split=0.3, split_strat=1, image_shape=(80, 80, 3)):

        print("Data loader initialized")
        self.data_path = data_path
        self.n_classes = 0
        self.data_type = data_type
        self.image_shape = image_shape
        self.class_labels = []
        self.get_class_data()
        self.split = split
        self.split_strat = split_strat
        random.seed(23)


    def default_split(self, data):
        train_index = int(len(data) * (1 - self.split))
        train = data[: train_index]
        test = data[train_index:]
        return train, test

    def file_split(self, class_name, data):
        df = pd.read_csv("{}splits/{}_test_split1.txt".format(self.data_path,class_name), sep=" ", header=None)
        videos = [d[2] for d in data]
        train = []
        test = []
        counter = 0
        for video in videos:
            text = "{}.avi".format(video)
            key = df[df[0] == text]
            if len(key) > 0:
                index = key.index[0]
                if df[1][index] == 2:
                    test.append(data[counter])
                elif df[1][index] == 1:
                    train.append(data[counter])
            else:
                train.append(data[counter])

            counter += 1

        return train, test

    def load_data(self):

        train = []
        test = []
        print("Loading data into memory...")
        pbar = tqdm(total=self.n_classes)
        if self.data_type == "frames":
            listing = os.listdir("{}{}/".format(self.data_path, "frames"))
            for class_name in listing:
                listing2 = os.listdir("{}{}/{}/".format(self.data_path, "frames", class_name))
                data = []
                for video in listing2:
                    listing3 = os.listdir("{}{}/{}/{}".format(self.data_path, "frames", class_name, video))
                    frames = []
                    for frame in listing3:
                        frame_path = "{}{}/{}/{}/{}".format(self.data_path, "frames", class_name, video, frame)
                        frames.append(frame_path)
                    sequence = self.build_image_sequence(frames)
                    label = self.get_class_one_hot(class_name)
                    data.append((sequence, label, video))

                if self.split_strat == 1:
                    a, b = self.default_split(data)
                    [train.append(t) for t in a]
                    [test.append(t) for t in b]

                if self.split_strat == 2:
                    a, b = self.file_split(class_name, data)
                    [train.append(t) for t in a]
                    [test.append(t) for t in b]

                pbar.update(1)

        elif self.data_type == "features":
            listing = os.listdir("{}{}/".format(self.data_path, "features"))
            for class_name in listing:
                listing2 = os.listdir("{}{}/{}/".format(self.data_path, "features", class_name))
                data = []
                for video in listing2:
                    np_path = "{}{}/{}/{}".format(self.data_path, "features", class_name, video)
                    sequence = np.load(np_path)
                    label = self.get_class_one_hot(class_name)
                    data.append((sequence, label, video[:-4]))

                if self.split_strat == 1:
                    a, b = self.default_split(data)
                    [train.append(t) for t in a]
                    [test.append(t) for t in b]

                if self.split_strat == 2:
                    a, b = self.file_split(class_name, data)
                    [train.append(t) for t in a]
                    [test.append(t) for t in b]

                pbar.update(1)

        pbar.close()

        random.shuffle(train)
        random.shuffle(test)

        X = [t[0] for t in train]
        y = [t[1] for t in train]
        X_test = [t[0] for t in test]
        y_test = [t[1] for t in test]
        return np.array(X), np.array(y), np.array(X_test), np.array(y_test), self.n_classes

    def get_class_data(self):
        listing = os.listdir("{}{}/".format(self.data_path,"frames"))
        class_labels = []
        class_count = 0
        for c in listing:
            class_labels.append(c)
            class_count += 1

        self.n_classes = class_count
        self.class_labels = class_labels

    def get_class_one_hot(self, class_str):
        # hot encoding
        label_encoded = self.class_labels.index(class_str)
        label_hot = to_categorical(label_encoded, self.n_classes)
        assert len(label_hot) == self.n_classes
        return label_hot

    # Given a set of frames (file paths), build our sequence.
    def build_image_sequence(self, frames):
        return [self.process_image(x, self.image_shape) for x in frames]

    # Given an image, process it and return the array
    @staticmethod
    def process_image(image, target_shape):
        # Load the image.
        h, w, _ = target_shape
        image = load_img(image, target_size=(h, w))

        # Turn it into numpy, normalize and return.
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)
        return x


