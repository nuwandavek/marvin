import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.01):
    return tf.compat.v1.maximum(alpha * x, x)

def create_cell(dim, n_layers, dropout):
    cell = tf.compat.v1.nn.rnn_cell.GRUCell(dim)
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    if n_layers > 1:
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    return cell

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def linear(inp, dim_out, scope, reuse=False):
    dim_in = inp.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.compat.v1.get_variable('W', [dim_in, dim_out])
        b = tf.compat.v1.get_variable('b', [dim_out])
    return tf.compat.v1.matmul(inp, W) + b

def combine(x, y, scope, reuse=False):
    dim_x = x.get_shape().as_list()[-1]
    dim_y = y.get_shape().as_list()[-1]

    with tf.compat.v1.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.compat.v1.get_variable('W', [dim_x+dim_y, dim_x])
        b = tf.compat.v1.get_variable('b', [dim_x])

    h = tf.compat.v1.matmul(tf.compat.v1.concat([x, y], 1), W) + b
    return leaky_relu(h)

def feed_forward(inp, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]

    with tf.compat.v1.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W1 = tf.compat.v1.get_variable('W1', [dim, dim])
        b1 = tf.compat.v1.get_variable('b1', [dim])
        W2 = tf.compat.v1.get_variable('W2', [dim, 1])
        b2 = tf.compat.v1.get_variable('b2', [1])
    h1 = leaky_relu(tf.compat.v1.matmul(inp, W1) + b1)
    logits = tf.compat.v1.matmul(h1, W2) + b2

    return tf.compat.v1.reshape(logits, [-1])

def gumbel_softmax(logits, gamma, eps=1e-20):
    U = tf.compat.v1.random_uniform(tf.compat.v1.shape(logits))
    G = -tf.compat.v1.log(-tf.compat.v1.log(U + eps) + eps)
    return tf.compat.v1.nn.softmax((logits + G) / gamma)

def softsample_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.compat.v1.nn.dropout(output, dropout)
        logits = tf.compat.v1.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.compat.v1.matmul(prob, embedding)
        return inp, logits

    return loop_func

def softmax_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.compat.v1.nn.dropout(output, dropout)
        logits = tf.compat.v1.matmul(output, proj_W) + proj_b
        prob = tf.compat.v1.nn.softmax(logits / gamma)
        inp = tf.compat.v1.matmul(prob, embedding)
        return inp, logits

    return loop_func

def argmax_word(dropout, proj_W, proj_b, embedding):

    def loop_func(output):
        output = tf.compat.v1.nn.dropout(output, dropout)
        logits = tf.compat.v1.matmul(output, proj_W) + proj_b
        word = tf.compat.v1.argmax(logits, axis=1)
        inp = tf.compat.v1.nn.embedding_lookup(embedding, word)
        return inp, logits

    return loop_func

def rnn_decode(h, inp, length, cell, loop_func, scope):
    h_seq, logits_seq = [], []

    with tf.compat.v1.variable_scope(scope):
        tf.compat.v1.get_variable_scope().reuse_variables()
        for t in range(length):
            h_seq.append(tf.compat.v1.expand_dims(h, 1))
            output, h = cell(inp, h)
            inp, logits = loop_func(output)
            logits_seq.append(tf.compat.v1.expand_dims(logits, 1))

    return tf.compat.v1.concat(h_seq, 1), tf.compat.v1.concat(logits_seq, 1)

def cnn(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]
    inp = tf.compat.v1.expand_dims(inp, -1)

    with tf.compat.v1.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.compat.v1.variable_scope('conv-maxpool-%s' % size):
                W = tf.compat.v1.get_variable('W', [size, dim, 1, n_filters])
                b = tf.compat.v1.get_variable('b', [n_filters])
                conv = tf.compat.v1.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.compat.v1.compat.v1.reduce_max(h, reduction_indices=1)
                pooled = tf.compat.v1.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.compat.v1.concat(outputs, 1)
        outputs = tf.compat.v1.nn.dropout(outputs, dropout)

        with tf.compat.v1.variable_scope('output'):
            W = tf.compat.v1.get_variable('W', [n_filters*len(filter_sizes), 1])
            b = tf.compat.v1.get_variable('b', [1])
            logits = tf.compat.v1.reshape(tf.compat.v1.matmul(outputs, W) + b, [-1])

    return logits

def discriminator(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    d_real = cnn(x_real, filter_sizes, n_filters, dropout, scope)
    d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)

    if wgan:
        eps = tf.compat.v1.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.compat.v1.gradients(d_mix, mix)[0]
        grad_norm = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.compat.v1.square(grad_norm-1)
        return tf.compat.v1.reduce_mean(loss), -tf.compat.v1.reduce_mean(loss)

    else:
        loss_d = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real)) + \
                 tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))
        loss_g = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_fake))
        return loss_d, loss_g
