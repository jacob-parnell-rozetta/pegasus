import tensorflow as tf


def ffn_model(features):
    # should the input be the hidden states rather than inputs from data?
    ffn_input_size = features["inputs"].get_shape().as_list()[1]  # max input length 512
    # ffn_target_size = features["targets"].get_shape().as_list()[1]  # max target length 32
    hidden1_size = 128
    ffn_output_size = 1  # scalar value to subtract from rouge loss

    # x = tf.placeholder(tf.float32, [None, ffn_input_size], name='data')
    # y = tf.placeholder(tf.float32, [None, ffn_target_size], name='targets')

    weights = {"w1": tf.Variable(tf.random_normal([ffn_input_size, hidden1_size]) * 0.01),
               "w2": tf.Variable(tf.random_normal([hidden1_size, ffn_output_size]) * 0.01)}

    biases = {"b1": tf.Variable(tf.zeros([hidden1_size])),
              "b2": tf.Variable(tf.zeros([ffn_output_size]))}

    def shallow_network(x_input, w, b):
        layer_1 = tf.add(tf.matmul(x_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])
        return output_layer

    with tf.variable_scope("control_variate", reuse=tf.AUTO_REUSE):
        ffn_output = shallow_network(features["inputs"], weights, biases)  # baseline scorer

    return ffn_output
