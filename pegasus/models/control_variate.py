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


def Q_func(z, target):
    combined = tf.concat([z, target], axis=1)  # concat([BxTxV, BxTxV], 1) -> [Bx2Tx2V]

    h1 = tf.layers.dense(2. * combined - 1., 1024, tf.nn.relu, name="q_1", use_bias=True)
    # h2 = tf.layers.dense(h1, 10, tf.nn.relu, name="q_2", use_bias=True)
    out = tf.layers.dense(h1, 1, name="q_out", use_bias=True)  # [Bx2Tx1]

    # [Bx2Tx1] -> [Bx2T] -> [BxT]
    flat_out = tf.squeeze(out, 2)
    flat_out = tf.layers.dense(flat_out, target.get_shape().as_list()[1], None, name="q_flat")

    norm_out = -tf.nn.sigmoid(flat_out)  # sigmoid for normalisation, minus for correct range
    return norm_out


def rwb_Q_func(pred, target):
    # assuming the target is BxT and pred is also BxT -> the logps
    hs_len = target.get_shape().as_list()[-1]

    h1_pred = tf.compat.v1.layers.dense(pred, hs_len, tf.nn.relu, name="q_pred", use_bias=True)
    h1_target = tf.compat.v1.layers.dense(target, hs_len, tf.nn.relu, name="q_target", use_bias=True)

    # concatenate
    combined = tf.concat([h1_pred, h1_target], axis=1)
    out = tf.compat.v1.layers.dense(combined, 1, name="q_out", use_bias=False)  # [1, 1]

    norm_out = -tf.nn.sigmoid(out)  # sigmoid for normalisation, minus for correct range
    return tf.squeeze(tf.squeeze(norm_out, -1), -1)  # returns as [1,1] -> [] scalar
