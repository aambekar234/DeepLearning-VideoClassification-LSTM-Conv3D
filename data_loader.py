import os
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


class DataLoader:

    def __init__(self, data_path, data_type, image_shape=(80, 80, 3)):

        print("Data loader initialized")
        self.data_path = data_path
        self.n_classes = 0
        self.data_type = data_type
        self.image_shape = image_shape
        self.class_labels = []
        self.get_class_data()

    def load_data(self, data_type=None, split=0.3):

        if data_type is not None:
            self.data_type = data_type

        train = []
        test = []
        print("Loading data into memory...")
        pbar = tqdm(total=self.n_classes)
        if self.data_type == "frames":
            listing = os.listdir("{}{}/".format(self.data_path, "frames"))
            for class_name in listing:
                listing2 = os.listdir("{}{}/{}/".format(self.data_path, "frames", class_name))
                video_id = 0
                total_videos = len(listing2)
                split_flag = (1.0 - split) * total_videos
                for video in listing2:
                    video_id += 1
                    listing3 = os.listdir("{}{}/{}/{}".format(self.data_path, "frames", class_name, video))
                    frames = []
                    for frame in listing3:
                        frame_path = "{}{}/{}/{}/{}".format(self.data_path, "frames", class_name, video, frame)
                        frames.append(frame_path)
                    sequence = self.build_image_sequence(frames)
                    label = self.get_class_one_hot(class_name)
                    if video_id <= split_flag:
                        train.append((sequence, label))
                    else:
                        test.append((sequence, label))
                pbar.update(1)

        elif self.data_type == "features":
            listing = os.listdir("{}{}/".format(self.data_path, "features"))
            for class_name in listing:
                listing2 = os.listdir("{}{}/{}/".format(self.data_path, "features", class_name))
                video_id = 0
                total_videos = len(listing2)
                split_flag = (1.0 - split) * total_videos
                for video in listing2:
                    video_id += 1
                    np_path = "{}{}/{}/{}".format(self.data_path, "features", class_name, video)
                    sequence = np.load(np_path)
                    label = self.get_class_one_hot(class_name)
                    if video_id <= split_flag:
                        train.append((sequence, label))
                    else:
                        test.append((sequence, label))
                pbar.update(1)

        pbar.close()

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

