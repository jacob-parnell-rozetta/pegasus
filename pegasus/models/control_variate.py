"""These are the varying control_variate functions we could implement for experiments."""
import tensorflow as tf


def ffn_baseline(inputs, ground_truth):
    ffn_input_size = inputs.get_shape().as_list()[2]  # max input length [x,y,z]
    ffn_target_size = ground_truth.get_shape().as_list()[2]  # hidden state size 512 [B, I, D]
    hidden1_size = 128
    hidden2_size = 16
    ffn_output_size = 1  # scalar value to subtract from rouge loss

    # Names used to extract trainable variables in RELAX variance reduction
    weights = {"w1": tf.Variable(tf.random_normal([ffn_input_size, hidden1_size]) * 0.01, name="control_variate_w1"),
               "w2": tf.Variable(tf.random_normal([hidden1_size, ffn_output_size]) * 0.01, name="control_variate_w2"),
               "w3": tf.Variable(tf.random_normal([ffn_target_size, hidden2_size]) * 0.01, name="control_variate_w3"),
               "w4": tf.Variable(tf.random_normal([hidden2_size, ffn_output_size]) * 0.01, name="control_variate_w4")}

    biases = {"b1": tf.Variable(tf.zeros([hidden1_size]), name="control_variate_b1"),
              "b2": tf.Variable(tf.zeros([ffn_output_size]), name="control_variate_b2"),
              "b3": tf.Variable(tf.zeros([hidden2_size]), name="control_variate_b3"),
              "b4": tf.Variable(tf.zeros([ffn_output_size]), name="control_variate_b4")}

    def shallow_network_baseline(x_input, y_input, w, b):
        layer_1 = tf.add(tf.matmul(x_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer_1 = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])

        layer_2 = tf.add(tf.matmul(y_input, w['w3']), b['b3'])
        layer_2 = tf.nn.relu(layer_2)
        output_layer_2 = tf.add(tf.matmul(layer_2, w['w4']), b['b4'])

        # take the last two elements from either encoded layer (reduce sum or reduce mean)  # TODO: FIX?
        combine = tf.reduce_sum(tf.stack([tf.squeeze(tf.squeeze(output_layer_1, axis=2), axis=0)[-1],
                                          tf.squeeze(tf.squeeze(output_layer_2, axis=2), axis=0)[-1]]))
        return combine

    with tf.variable_scope("ffn_baseline", reuse=tf.AUTO_REUSE):
        ffn_output = shallow_network_baseline(inputs, ground_truth, weights, biases)

    return ffn_output


def control_variate(input):
    ffn_input_size = input.get_shape().as_list()[2]
    hidden1_size = 128
    ffn_output_size = 1

    # Names used to extract trainable variables in RELAX variance reduction
    weights = {"w1": tf.Variable(tf.random_normal([ffn_input_size, hidden1_size]) * 0.01, name="control_variate_w1"),
               "w2": tf.Variable(tf.random_normal([hidden1_size, ffn_output_size]) * 0.01, name="control_variate_w2")}

    biases = {"b1": tf.Variable(tf.zeros([hidden1_size]), name="control_variate_b1"),
              "b2": tf.Variable(tf.zeros([ffn_output_size]), name="control_variate_b2")}

    def shallow_network_relax(x_input, w, b):
        layer_1 = tf.add(tf.matmul(x_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer_1 = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])

        combine = tf.reduce_mean(output_layer_1)  # reduce_mean in RELAX?
        return combine

    with tf.variable_scope("control_variate", reuse=tf.AUTO_REUSE):
        control_variate_output = shallow_network_relax(input, weights, biases)

    return control_variate_output


def Q_func(z):
    h1 = tf.layers.dense(2. * z - 1., 64, tf.nn.tanh, name="q_1", use_bias=True)  # 64, and ELU in paper
    # h2 = tf.layers.dense(h1, 10, tf.nn.relu, name="q_2", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)
    out = tf.nn.softmax(out)
    return tf.reduce_mean(out)  # should this mimic ROUGE score, or ROUGE loss? (+/-)
