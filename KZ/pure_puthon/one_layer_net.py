from pure_python.neuron import Neuron
from math import pow


class OneLayerNet:

    def __init__(self, inputs_count, output_neurons_count): #количество входо, количество выходых нейронов
        self.__inputs_count = inputs_count
        self.__neurons = []
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(inputs_count))


    def train(self, vector, learning_rate):

        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x()) # из вектора берем обучающий  вектор ,calc_y-вычислял веса в нейроне

      
        weights_deltas = [[0] * (len(vector.get_x()) + 1)] * len(self.__neurons)   #вычисляем вес для дельты

        for j in range(len(self.__neurons)): #перебираем нейроны
            sigma = (vector.get_d()[j] - self.__neurons[j].get_y()) \#считаем для каждого нейрона сигму
                    * self.__neurons[j].derivative()
            weights_deltas[j][0] = learning_rate * sigma #нулевой строке массива дельта весов присваиваем значение 
            wlen = len(self.__neurons[j].get_weights())#количество весов у контретного нейрона
            for i in range(wlen): #проходим по всем весам каждой строки
                weights_deltas[j][i] = learning_rate * sigma * vector.get_x()[i]#считаем новые веса для каждого элемента массива, например для первой
                #строки умножаем на первое значение обучающего вектора, для второй строки и тд
            self.__neurons[j].correct_weights(weights_deltas[j]) #для каждого нейрона корректируем вес тем,что прибавляем получившуюся дельту вусов

        loss = 0
        # шаг 5 вычисляем сумму  ошибок для каждого нейрона
        for j in range(len(self.__neurons)):
            loss += pow(vector.get_d()[j] - self.__neurons[j].get_y(), 2)

#искусственно уменьшаем ошибку?
        return 0.5 * loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x()) # из вектора берем обучающий  вектор ,calc_y-вычислял веса в нейроне

            y[j] = self.__neurons[j].get_y()
        return y




















