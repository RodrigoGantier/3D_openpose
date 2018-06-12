from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf


def kaiming(shape, dtype, partition_info=None):
    return (tf.truncated_normal(shape, dtype=dtype) * tf.sqrt(2 / float(shape[0])))


def lineal_model(human_points, dropout_keep_prob, istraining=True):

    with vs.variable_scope("Zero_layer"):
        w0 = tf.get_variable(name="w0", initializer=kaiming, shape=[14 * 2, 1024], dtype=tf.float32)
        b0 = tf.get_variable(name="b0", initializer=kaiming, shape=[1024], dtype=tf.float32)
        w0 = tf.clip_by_norm(w0, 1)
        y0 = tf.matmul(human_points, w0) + b0

        y0 = tf.layers.batch_normalization(y0, training=istraining, name="batch_normalization_0")
        y0 = tf.nn.relu(y0)
        y0 = tf.nn.dropout(y0, dropout_keep_prob)

    with vs.variable_scope("layer_one"):
        w1 = tf.get_variable(name="w1", initializer=kaiming, shape=[1024, 1024], dtype=tf.float32)
        b1 = tf.get_variable(name="b1", initializer=kaiming, shape=[1024], dtype=tf.float32)
        w1 = tf.clip_by_norm(w1, 1)
        y1 = tf.matmul(y0, w1) + b1

        y1 = tf.layers.batch_normalization(y1, training=istraining, name="batch_normalization_1")
        y1 = tf.nn.relu(y1)
        y1 = tf.nn.dropout(y1, dropout_keep_prob) + y0

    with vs.variable_scope("layer_two"):
        w2 = tf.get_variable(name="w2", initializer=kaiming, shape=[1024, 1024], dtype=tf.float32)
        b2 = tf.get_variable(name="b2", initializer=kaiming, shape=[1024], dtype=tf.float32)
        w2 = tf.clip_by_norm(w2, 1)
        y2 = tf.matmul(y1, w2) + b2

        y2 = tf.layers.batch_normalization(y2, training=istraining, name="batch_normalization_2")
        y2 = tf.nn.relu(y2)
        y2 = tf.nn.dropout(y2, dropout_keep_prob) + y1

    with vs.variable_scope("layer_output"):
        w3 = tf.get_variable(name="w3", initializer=kaiming, shape=[1024, 14], dtype=tf.float32)
        b3 = tf.get_variable(name="b3", initializer=kaiming, shape=[14], dtype=tf.float32)
        w3 = tf.clip_by_norm(w3, 1)
        y3 = tf.nn.bias_add(tf.matmul(y2, w3), b3)

        # y3 = tf.layers.batch_normalization(y3, training=istraining, name="batch_normalization_1")
        # y3 = tf.nn.relu(y3)
        # y3 = tf.nn.dropout(y3, dropout_keep_prob)
    return y3
