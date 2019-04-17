from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import numpy as np
import cv2
import os
from concurrent.futures.thread import ThreadPoolExecutor
import shutil
import math
from tqdm import tqdm


class DataGen:
    def __init__(self, data_path, op_path=None, fpv=30, image_size=(299, 299), weights=None):
        """load imagenet weights for extractions"""

        if op_path is None:
            op_path = "{}_op_{}/".format(data_path[:-1], fpv)

        self.weights = weights  # so we can check elsewhere which model
        self.data_path = data_path  # dir containing video dirs
        self.op_path = op_path  # dir where all data will be generated
        self.fpv = fpv  # frame needs to be extracted per video file
        self.image_size = image_size # image size of extracted frame
        self.op_frames_path = "{}{}/".format(op_path, "frames")
        self.op_features_path = "{}{}/".format(op_path, "features")

        if weights is None:
            print("Skipping model load")
            self.model = None
            # self.load_inception_model()

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def load_inception_model(self):
        # Get model with pre-trained weights.
        base_model = InceptionV3(weights='imagenet', include_top=True)

        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    # removes all generated files and recreates directories
    def clean_project(self):
        if os.path.isdir(self.op_path):
            shutil.rmtree(self.op_path)

        os.makedirs(self.op_path)
        os.makedirs(self.op_features_path)
        os.makedirs(self.op_frames_path)

    # Extracts frames from video source of class
    # class_path : path to video data class
    # op_path : output path for the extracted frames
    def frame_extractor(self, class_path, op_path, pbar):
        listing = os.listdir(class_path)
        count = 1

        for file in listing:
            video = cv2.VideoCapture("{}{}".format(class_path, file))
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            divider = math.floor(length / self.fpv)
            if divider > 0:
                frames_dir_name = "{}video_{}".format(op_path, count)
                os.makedirs(frames_dir_name)
                image_id = 1
                while video.isOpened():
                    frameId = video.get(1)
                    status, frame = video.read()
                    if status:
                        frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
                    else:
                        break
                    if frameId % divider == 0:
                        filename = "{}/image_{}.jpg".format(frames_dir_name, image_id)
                        if image_id <= self.fpv:
                            cv2.imwrite(filename, frame)
                        image_id += 1
                video.release()
                count += 1
        pbar.update(1)

    # function which generates frame data and feature data
    def generate_data(self):
        print("Please wait.. Extracting frames")
        class_count, video_count = self.get_pbar_length()
        pbar1 = tqdm(total=class_count)

        self.clean_project()
        listing = os.listdir(self.data_path)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for d in listing:
                class_path = "{}{}/".format(self.data_path, d)
                op_path = "{}{}/".format(self.op_frames_path, d)
                executor.submit(self.frame_extractor, class_path, op_path, pbar1)

        print("Frame extraction completed")
        pbar1.close()

        pbar2 = tqdm(total=(video_count * self.fpv))
        print("Feature extraction started...")
        self.save_video_features(self.op_frames_path, self.op_features_path, pbar2)
        print("Feature extraction completed")
        pbar2.close()

    # get numbers for loading UI
    def get_pbar_length(self):
        listing = os.listdir(self.data_path)
        class_count = 0
        video_count = 0
        for c in listing:
            class_count += 1
            listing2 = os.listdir("{}{}/".format(self.data_path,c))
            for _ in listing2:
                video_count += 1

        return class_count, video_count

    # reads image, extracts features
    def feature_extractor(self, image_path):
        if self.model is None:
            self.load_inception_model()

        x = cv2.imread(image_path)

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

    # extracts features of images in a dir and saves as numpy array on disk
    def save_video_features(self, in_path, op_path, pbar):

        listing = os.listdir(in_path)

        for video_c in listing:
            video_c_dir = "{}{}/".format(in_path,video_c)
            video_c_op_dir = "{}{}/".format(op_path,video_c)
            os.mkdir(video_c_op_dir)
            listing2 = os.listdir(video_c_dir)
            for video_dir in listing2:
                frames_dir = "{}{}/".format(video_c_dir, video_dir)
                listing3 = os.listdir(frames_dir)
                sequence = []
                for frame in listing3:
                    frame_path = "{}{}".format(frames_dir, frame)
                    features = self.feature_extractor(frame_path)
                    sequence.append(features)
                    pbar.update(1)

                op_feature_file_name = "{}/{}.npy".format(video_c_op_dir, video_dir)
                np.save(op_feature_file_name, sequence)


model = DataGen("hmdb/")
model.generate_data()
