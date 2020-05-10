from pure_python.neuron import Neuron
from math import pow


class OneLayerNet:

    def __init__(self, inputs_count, output_neurons_count):  #объявление кол-ва входов и выходных нейронов.
        self.__inputs_count = inputs_count
        self.__neurons = []
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(inputs_count))


    def train(self, vector, learning_rate):

        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x()) # из вектора берем обучающий  вектор ,calc_y-вычислял веса в нейроне
        # вес дельты
        weights_deltas = [[0] * (len(vector.get_x()) + 1)] * len(self.__neurons)

        for j in range(len(self.__neurons)):
            sigma = (vector.get_d()[j] - self.__neurons[j].get_y()) \
                    * self.__neurons[j].derivative()
            weights_deltas[j][0] = learning_rate * sigma
            wlen = len(self.__neurons[j].get_weights())
            for i in range(wlen):
                weights_deltas[j][i] = learning_rate * sigma * vector.get_x()[i]
            self.__neurons[j].correct_weights(weights_deltas[j])

        loss = 0
        # шаг 5 вычисляем ошибку для каждого нейрона
        for j in range(len(self.__neurons)):
            loss += pow(vector.get_d()[j] - self.__neurons[j].get_y(), 2)
        # искусственно берем пол ошибки?
        return 0.5 * loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x()) # из вектора берем обучающий  вектор ,calc_y-вычислял веса в нейроне
            y[j] = self.__neurons[j].get_y()
        return y




















