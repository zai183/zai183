from tensor_flow import neural_util as nu


class OneLayerNet(object):

    def __init__(self, x, num_classes): #функция, преобразующая входные аргументы x, num_classes в параметры класса


        self.X = x
        self.NUM_CLASSES = num_classes

        #задаем парметр/функцию для построения графа
        self.output = self.create()

    def create(self):  #функция постройки графа
        return nu.fc(self.X, self.X.get_shape()[1], self.NUM_CLASSES, name='one_layer_perceptron')