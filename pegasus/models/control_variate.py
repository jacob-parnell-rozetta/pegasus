import tensorflow as tf


def ffn_model(inputs, ground_truth):
    # TODO: pass the inputs AND ground_truth with two encoders, and an output layer that generates a single score
    ffn_input_size = inputs.get_shape().as_list()[2]  # max input length [x,y,z]
    # ffn_target_size = features["targets"].get_shape().as_list()[1]  # max target length 32
    hidden1_size = 128
    ffn_output_size = 1  # scalar value to subtract from rouge loss

    # x = tf.placeholder(tf.float32, [None, ffn_input_size], name='data')
    # y = tf.placeholder(tf.float32, [None, ffn_target_size], name='targets')

    # Names used to extract trainable variables in RELAX variance reduction
    weights = {"w1": tf.Variable(tf.random_normal([ffn_input_size, hidden1_size]) * 0.01, name="control_variate_w1"),
               "w2": tf.Variable(tf.random_normal([hidden1_size, ffn_output_size]) * 0.01, name="control_variate_w2")}

    biases = {"b1": tf.Variable(tf.zeros([hidden1_size]), name="control_variate_b1"),
              "b2": tf.Variable(tf.zeros([ffn_output_size]), name="control_variate_b2")}

    def shallow_network(x_input, w, b):
        # x_input = tf.cast(x_input, tf.float32)  # convert inputs to same type as weights
        layer_1 = tf.add(tf.matmul(x_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])
        return output_layer

    with tf.variable_scope("control_variate", reuse=tf.AUTO_REUSE):
        ffn_output = shallow_network(inputs, weights, biases)  # baseline scorer

    return ffn_output
