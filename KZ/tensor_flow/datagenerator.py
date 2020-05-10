import numpy as np
import cv2


class DataGenerator:
    def __init__(self, patterns, labels, scale_size, shuffle=False, input_channels=3, nb_classes=8):

        # Init params
        self.__n_classes = nb_classes
        self.__shuffle = shuffle
        self.__input_channels = input_channels
        self.__scale_size = scale_size
        self.__pointer = 0
        self.__data_size = len(labels)
        self.__patterns = patterns
        self.__labels = labels
        
        if self.__shuffle:
            self.shuffle_data()

    def get_data_size(self):
        return self.__data_size

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.__patterns.copy()
        labels = self.__labels.copy()
        self.__patterns = []
        self.__labels = []
        
        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.__patterns.append(images[i])
            self.__labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.__pointer = 0
        
        if self.__shuffle:
            self.shuffle_data()

    def next(self):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        path = self.__patterns[self.__pointer]
        label = self.__labels[self.__pointer]
        
        # update pointer
        self.__pointer += 1
        
        # Read pattern
        if self.__input_channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)

        # rescale image
        img = cv2.resize(img, (self.__scale_size[0], self.__scale_size[1]))
        img = img.astype(np.float32)

        if self.__input_channels == 1:
            img = img.reshape((img.shape[0], img.shape[1], 1)).astype(np.float32)

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros(self.__n_classes)
        one_hot_labels[label] = 1

        # return array of images and labels
        return img, one_hot_labels
