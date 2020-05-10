import os
from tensor_flow.one_layer_net_tf import OneLayerNet
import tensorflow as tf
from datareader import DataReader
from tensor_flow.vector_tf import Vector
from datetime import datetime


def get_max_neuron_idx(neurons):
    max_idx = -1
    answer = -1
    for j in range(len(neurons)):
        if neurons[j] > answer:
            answer = neurons[j]
            max_idx = j
    return max_idx


# Learning params
learning_rate = 1
num_epochs = 30

# Network params
input_channels = 1
input_height = 28
input_width = 28
num_classes = 9
save_histogram = False

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
log_path = "../tmp/log/"
filewriter_path = log_path + datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + "/"
checkpoint_path = "../tmp/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.compat.v1.placeholder(tf.float32, [1, input_height * input_width])
y = tf.compat.v1.placeholder(tf.float32, [None, num_classes])
learning_val = tf.placeholder(tf.float32, [], name='learning_rate')

# Initialize model
model = OneLayerNet(x, num_classes)

# Link variable to model output
score = model.output

# List of trainable variables of the layers we want to train
var_list = tf.trainable_variables()

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss_elem = tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=y)
    loss = tf.reduce_mean(loss_elem)

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

if save_histogram:
    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

valid_summary = tf.Summary()

# Initialize the FileWriter
writer_1 = tf.summary.FileWriter(filewriter_path + 'train')
writer_2 = tf.summary.FileWriter(filewriter_path + 'validation')

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

train_dir = 'data/train'
test_dir = 'data/test'

train_generator = DataReader(train_dir, [input_height, input_width], True, input_channels, num_classes).get_generator()
test_generator = DataReader(test_dir, [input_height, input_width], False, input_channels, num_classes).get_generator()

print('Size of training set: {}'.format(train_generator.get_data_size()))
print('Size of testing set: {}'.format(test_generator.get_data_size()))

train_patterns_per_epoch = train_generator.get_data_size()

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer_1.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard by command: tensorboard --logdir {}".format(datetime.now(),
                                                      log_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_patterns_per_epoch:
            xs, ds = train_generator.next()
            vector = Vector(xs, ds)
            xs = vector.get_x()
            ds = vector.get_desire_outputs()

            # And run the training op
            sess.run(train_op, feed_dict={x: xs,
                                          y: ds,
                                          learning_val: learning_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: xs,
                                                        y: ds})
                writer_1.add_summary(s, epoch * train_patterns_per_epoch + step)

            step += 1

        train_generator.reset_pointer()
        '''
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        '''
    test_patterns_count = test_generator.get_data_size()

    # Test the model on the entire test set
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
