from one_layer_net import OneLayerNet
from datareader import DataReader
from _vector import Vector
from datetime import datetime
import numpy as np #NumPy — библиотека с открытым исходным кодом для языка программирования Python.
# Возможности: поддержка многомерных массивов;
# поддержка высокоуровневых математических функций, предназначенных для работы с многомерными массивами
import cv2 #библ  Opencv для машинного обучения, можно: размер, поворот,градация, размытие,рисование и тд


def get_max_neuron_idx(neurons): #получить максимальный нейрон
    max_idx = -1
    answer = -1
    for j in range(len(neurons)): # len (ф)-в диапазоне ф
        # range() позволяет вам генерировать ряд чисел в рамках заданного диапазона.
        if neurons[j] > answer: # в теории [j] номер нейрона
            answer = neurons[j]
            max_idx = j
    return max_idx


# скоростные параметры
learning_rate = 1e-6 #скорость обучения
num_epochs = 10 #количество эпох

# Network params
input_channels = 1 #входной канал
input_height = 28 #высота
input_width = 28 #ширина
num_classes = 6 #количество классов изображений

one_layer_net = OneLayerNet(input_height * input_width, num_classes)  #размер изображения, класс изображения(двойkа , тройка и тд)

#путь к обучающим файлам
train_dir = "data/train"
test_dir = "data/test"

train_generator = DataReader(train_dir, [input_height, input_width], True, input_channels, num_classes).get_generator()
#берем с помощью DataReader изображения, размер, тру?,
test_generator = DataReader(test_dir, [input_height, input_width], False, input_channels, num_classes).get_generator()

print('Size of training set: {}'.format(train_generator.get_data_size())) #объем тренировочных изображений
print('Size of testing set: {}'.format(test_generator.get_data_size()))#объем тесовых изображенй изображений


print("{} Start training...".format(datetime.now())) # текущее время начала обучения

# Подготовить обучающую выборку, каждый элемент которой будет
# состоять из пар (X, D)m (m=1,…q) – обучающего вектора X = (x1,…,xn) (i=1,…,n) с
# вектором желаемых значений D = (d1,…,dk) (j=1,…,k) выходов персептрона.
for epoch in range(num_epochs):
    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
    loss = 0
    for m in range(train_generator.get_data_size()):
        x, d = train_generator.next()
        loss += one_layer_net.train(Vector(x, d), learning_rate)
    print("loss = {}".format(loss / train_generator.get_data_size()))
    train_generator.reset_pointer()
    train_generator.shuffle_data()

passed = 0
for i in range(test_generator.get_data_size()): # get_data_size получить размер данных
    x, d = test_generator.next()
    y = one_layer_net.test(Vector(x, d))

    d_max_idx = get_max_neuron_idx(d)
    y_max_idx = get_max_neuron_idx(y)
    if y_max_idx == d_max_idx:
        passed += 1
    print("{} recognized as {}".format(d_max_idx, y_max_idx))

accuracy = passed / test_generator.get_data_size() * 100.0
print("Accuracy: {:.4f}%".format(accuracy))

print("Recognizing custom image") #Распознавание собственного изображения
img = cv2.imread("custom.bmp", cv2.IMREAD_GRAYSCALE) #custom.bmp -входное изображение ,которым определяется,как хорошо нейровна распознает
# cv2.imread-импортирует изображение и просматривает , в кавычках путь
img = img.reshape((img.shape[0], img.shape[1], 1)).astype(np.float32)
y = one_layer_net.test(Vector(img, None))
print("Custom image recognized as {}".format(get_max_neuron_idx(y))) #изображение определяется как ---например 3 .
# format(get_max_neuron_idx(y))-цифра которой определилось озображение