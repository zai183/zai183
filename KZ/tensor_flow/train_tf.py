import os
from tensor_flow.one_layer_net_tf import OneLayerNet
import tensorflow as tf #TensorFlow — открытая программная библиотека для машинного обучения,
# разработанная компанией Google для решения задач построения и тренировки нейронной сети с целью
# автоматического нахождения и классификации образов, достигая качества человеческого восприятия.
from datareader import DataReader
from tensor_flow.vector_tf import Vector
from datetime import datetime


def get_max_neuron_idx(neurons): #получить максимальный нейрон
    max_idx = -1
    answer = -1
    for j in range(len(neurons)): # len (ф)-в диапазоне ф
        # range() позволяет вам генерировать ряд чисел в рамках заданного диапазона.
        if neurons[j] > answer:  # в теории [j] номер нейрона
            answer = neurons[j]
            max_idx = j
    return max_idx

learning_rate = 1
num_epochs = 30


input_channels = 1
input_height = 28
input_width = 28
num_classes = 9
save_histogram = False #сохранить гистограмму

# Как часто мы хотим записать данные tf.summary на диск
display_step = 1

# Путь для tf.summary.FileWriter и для хранения контрольных точек модели
log_path = "../tmp/log/"
filewriter_path = log_path + datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + "/"
checkpoint_path = "../tmp/"

# Создать родительский путь, если он не существует
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# TF заполнитель для ввода и вывода графиков
x = tf.compat.v1.placeholder(tf.float32, [1, input_height * input_width])
y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])
learning_val = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')

# инициализировать модель
model = OneLayerNet(x, num_classes)


# Связать переменную с выходом модели
score = model.output

# Список обучаемых переменных слоев, которые мы хотим обучить
var_list = tf.trainable_variables()

# Оп для расчета потерь
with tf.name_scope("cross_ent"):
    loss_elem = tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=y)
    loss = tf.reduce_mean(loss_elem)

# поезд оп
with tf.name_scope("train"):
    # Получить градиенты всех обучаемых переменных
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Создать оптимизатор и применить градиентный спуск к обучаемым переменным
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

if save_histogram:
    #Добавить градиенты в сводку
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Добавьте переменные, которые мы обучаем, в резюме
    for var in var_list:
        tf.summary.histogram(var.name, var)

tf.summary.scalar('cross_entropy', loss)


# Оценка оп: Точность модели
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Добавьте точность в резюме
tf.summary.scalar('accuracy', accuracy)

# Объединить все резюме вместе
merged_summary = tf.summary.merge_all()

valid_summary = tf.Summary()

# Инициализируйте FileWriter
writer_1 = tf.summary.FileWriter(filewriter_path + 'train')
writer_2 = tf.summary.FileWriter(filewriter_path + 'validation')

# Инициализируйте заставку для контрольных точек модели чекпоинт
saver = tf.train.Saver()

train_dir = '../data/train'
test_dir = '../data/test'

train_generator = DataReader(train_dir, [input_height, input_width], True, input_channels, num_classes).get_generator()
test_generator = DataReader(test_dir, [input_height, input_width], False, input_channels, num_classes).get_generator()

print('Size of training set: {}'.format(train_generator.get_data_size()))
print('Size of testing set: {}'.format(test_generator.get_data_size()))

train_patterns_per_epoch = train_generator.get_data_size()

# начать сеанс
with tf.Session() as sess:
    # инициализация всех переменных
    sess.run(tf.global_variables_initializer())

    # добавит грфик модели в  TensorBoard
    writer_1.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard by command: tensorboard --logdir {}".format(datetime.now(),
                                                      log_path))

    #
    # Цикл по количеству эпох
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_patterns_per_epoch:
            xs, ds = train_generator.next()
            vector = Vector(xs, ds)
            xs = vector.get_x()
            ds = vector.get_desire_outputs()

            # И запустить учебный оп
            sess.run(train_op, feed_dict={x: xs,
                                          y: ds,
                                          learning_val: learning_rate})

            #
            # Создать сводку с текущей партией данных и записать в файл
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: xs,
                                                        y: ds})
                writer_1.add_summary(s, epoch * train_patterns_per_epoch + step)

            step += 1

        train_generator.reset_pointer()
        '''
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # сохранить чекпоинт модели
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        '''
    test_patterns_count = test_generator.get_data_size()

    #тестирование на тестовом наборе
    print("{} Start testing".format(datetime.now()))
    passed = 0
    for _ in range(test_patterns_count):
        xs, ds = test_generator.next()
        vector = Vector(xs, ds)
        xs = vector.get_x()
        ds = vector.get_desire_outputs()

        prediction = sess.run(tf.nn.sigmoid(model.output), feed_dict={x: xs})

        d_max_idx = get_max_neuron_idx(list(ds.reshape(ds[0].shape)))
        y_max_idx = get_max_neuron_idx(prediction[0])
        if y_max_idx == d_max_idx:
            passed += 1
        print("{} recognized as {}".format(d_max_idx, y_max_idx))

    accuracy = passed / test_generator.get_data_size() * 100.0
    print("Accuracy: {:.4f}%".format(accuracy))
