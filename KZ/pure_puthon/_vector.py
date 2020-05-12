import numpy as np
#в общем класс возвращает параметры х и d в виде массива, внутри __init__делая его массивом нужного вида
class Vector:
    def __init__(self, x, d):
        if len(x.shape) > 2: #x.shape-размер массива
            self.__x = list(np.asanyarray(x).reshape(-2)) #asanyarray преобразует даннеы в массив
            # reshape- изменяет форму массива без изменения данных -2 это как?
        else:
            self.__x = x
        self.__d = d

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
