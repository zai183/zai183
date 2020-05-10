import numpy as np


#класс возвращает параметры x,desireOutputs в виде массива необходимого вида
class Vector:

    def __init__(self, x, desireOutputs):
        if len(x.shape) > 2:  #размер массива
            self.__x = np.asarray(x).reshape(-1)  #изменяет размерность массива, не меняя данные  внутри него
            self.__x = self.__x.reshape(1, self.__x.shape[0]).astype(np.float32)
        self.__desireOutputs = desireOutputs.reshape(1, desireOutputs.shape[0]) #изменяет размерность массива, не меняя данные  внутри него

    def get_x(self):
        return self.__x

    def get_desire_outputs(self):
        return self.__desireOutputs
