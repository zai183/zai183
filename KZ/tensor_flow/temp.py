import tensorflow as tf

#все это заполнитель, с помощью которого мы подаем в граф данные
# подробнее по https://databricks.com/tensorflow/placeholders
x = tf.placeholder("float", None)
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
