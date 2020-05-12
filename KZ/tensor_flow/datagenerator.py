import numpy as np
import cv2


class DataGenerator:
    def __init__(self, patterns, labels, scale_size, shuffle=False, input_channels=3, nb_classes=8):

        # Init params
        self.__n_classes = nb_classes
        self.__shuffle = shuffle  #перемешивание
        self.__input_channels = input_channels #входные сигналы
        self.__scale_size = scale_size #размер шкалы
        self.__pointer = 0  #указатель
        self.__data_size = len(labels)  #размер данных
        self.__patterns = patterns #шаблоны
        self.__labels = labels
        
        if self.__shuffle:
            self.shuffle_data()

    def get_data_size(self):
        return self.__data_size

    def shuffle_data(self): #Случайное перемешивание изображений и меток

        images = self.__patterns.copy()
        labels = self.__labels.copy()
        self.__patterns = []
        self.__labels = []
        
        # создать список перестановочного индекса и перемешать данные в соответствии с списком
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.__patterns.append(images[i])
            self.__labels.append(labels[i])
                
    def reset_pointer(self):  #сбросить указатель на начало списка

        self.__pointer = 0
        
        if self.__shuffle:
            self.shuffle_data()

    def next(self): #Эта функция получает следующие n (= batch_size) изображений из
        # списка путей и маркирует и загружает изображения в них в память

        # Получить следующую партию изображения (путь) и метки
        path = self.__patterns[self.__pointer]
        label = self.__labels[self.__pointer]
        
        # обновить указатель
        self.__pointer += 1

        # Читать шаблон
        if self.__input_channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)

        # масштабировать изображение. Масштаби́рование изображения — изменение размера изображения с сохранением пропорций.
        img = cv2.resize(img, (self.__scale_size[0], self.__scale_size[1]))
        img = img.astype(np.float32)

        if self.__input_channels == 1:
            img = img.reshape((img.shape[0], img.shape[1], 1)).astype(np.float32)

        # Развернуть метки до одной хот кодировки
        one_hot_labels = np.zeros(self.__n_classes) #Вернуть новый массив заданной формы и типа, заполненный нулями.nt  6
        one_hot_labels[label] = 1

        # вернуть массив изображений и меток
        return img, one_hot_labels
