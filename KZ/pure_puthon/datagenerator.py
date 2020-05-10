import numpy as np
import cv2
from numpy.core._multiarray_umath import ndarray


class DataGenerator:
    def __init__(self, patterns, labels, scale_size, shuffle=False, input_channels=3, nb_classes=5):

        # Инициация параметров
        self.__n_classes = nb_classes
        self.__shuffle = shuffle #перемешиваем данные
        self.__input_channels = input_channels #количество входных каналов/сиггналов
        self.__scale_size = scale_size #размер сетки/шкалы
        self.__pointer = 0 #указатель?
        self.__data_size = len(labels) #размер/длина данных
        self.__patterns = patterns #шаблоны/объекты
        self.__labels = labels
        
        if self.__shuffle:
            self.shuffle_data()

    def get_data_size(self):
        return self.__data_size

    def shuffle_data(self): #функция перемешивания исходных изображений и меток
        """
        Random shuffle the images and labels
        """
        images = self.__patterns.copy()
        labels = self.__labels.copy()
        self.__patterns = []
        self.__labels = []
        
        # create list of permutated index and shuffle data accoding to list
        # создается список, в соответствии с которым мешаются данные
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.__patterns.append(images[i])
            self.__labels.append(labels[i])
                
    def reset_pointer(self): # пермещение указателя в начало списка
        """
        reset pointer to begin of the list
        """
        self.__pointer = 0
        
        if self.__shuffle:
            self.shuffle_data()

    def next(self):  #Эта функция получает следующие n (= batch_size) изображений из
        # списка путей и маркирует и загружает изображения в них в память
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        # Получить следующую партию изображения (путь) и метки
        path = self.__patterns[self.__pointer]
        label = self.__labels[self.__pointer]
        

        # обновить указатель
        self.__pointer += 1
        
        # прочитать след. шаблон
        if self.__input_channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)

        # масштабирование изображения, но сохраняя пропорцию
        img = cv2.resize(img, (self.__scale_size[0], self.__scale_size[1]))
        img = img.astype(np.float32)

        # чтение шаблона
        if self.__input_channels == 1:
            img = img.reshape((img.shape[0], img.shape[1], 1)).astype(np.float32)

        # Expand labels to one hot encoding
        one_hot_labels: ndarray = np.zeros(self.__n_classes)
        one_hot_labels[label] = 1

        # вернуть массив с изображениями и метками в нем
        return img, one_hot_labels
