import tensorflow as tf


def ffn_model(inputs, ground_truth=None):
    ffn_input_size = inputs.get_shape().as_list()[2]  # max input length [x,y,z]
    # ffn_target_size = ground_truth.get_shape().as_list()[2]  # hidden state size 512 [B, I, D]
    hidden1_size = 128
    ffn_output_size = 1  # scalar value to subtract from rouge loss

    # x = tf.placeholder(tf.float32, [None, ffn_input_size], name='data')
    # y = tf.placeholder(tf.float32, [None, ffn_target_size], name='targets')

    # Names used to extract trainable variables in RELAX variance reduction
    weights = {"w1": tf.Variable(tf.random_normal([ffn_input_size, hidden1_size]) * 0.01, name="control_variate_w1"),
               "w2": tf.Variable(tf.random_normal([hidden1_size, ffn_output_size]) * 0.01, name="control_variate_w2")}

    biases = {"b1": tf.Variable(tf.zeros([hidden1_size]), name="control_variate_b1"),
              "b2": tf.Variable(tf.zeros([ffn_output_size]), name="control_variate_b2")}

    def shallow_network(x_input, w, b, y_input=None):
        layer_1 = tf.add(tf.matmul(x_input, w['w1']), b['b1'])
        layer_1 = tf.nn.relu(layer_1)
        output_layer_1 = tf.add(tf.matmul(layer_1, w['w2']), b['b2'])

        # If there is ground_truth (for reinforce_baseline)
        if y_input:
            layer_2 = tf.add(tf.matmul(y_input, w['w1']), b['b1'])
            layer_2 = tf.nn.relu(layer_2)
            output_layer_2 = tf.add(tf.matmul(layer_2, w['w2']), b['b2'])

            # reduce_mean to 'scalarify' the value -> output_layers seem to return shape same size as inputs
            combine = tf.concat([tf.squeeze(output_layer_1, axis=2), tf.squeeze(output_layer_2, axis=2)])
            return combine

        # For RELAX implementation
        else:
            return output_layer_1

    with tf.variable_scope("control_variate", reuse=tf.AUTO_REUSE):
        ffn_output = shallow_network(inputs, weights, biases, ground_truth)  # baseline scorer
        ffn_output = tf.reduce_mean(ffn_output)

    return ffn_output
