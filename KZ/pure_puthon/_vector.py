import numpy as np


#класс возвращает параметры x,d в виде массива необходимого вида
class Vector:
    def __init__(self, x, d):
        if len(x.shape) > 2: #размер массива
            self.__x = list(np.asanyarray(x).reshape(-2))  #asanyarray преобразует данные в массив
            # reshape- изменяет форму массива без изменения данных
        else:
            self.__x = x
        self.__d = d

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
