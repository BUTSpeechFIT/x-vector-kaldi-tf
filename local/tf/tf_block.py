import tensorflow as tf
from tensorflow.python.framework import ops


def __get_variable(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)


def batch_norm_wrapper(inputs, is_training, decay=0.99, epsilon=1e-3, name_prefix=''):
    gamma = __get_variable(name_prefix + 'gamma', inputs.get_shape()[-1], tf.constant_initializer(1.0))
    beta = __get_variable(name_prefix + 'beta', inputs.get_shape()[-1], tf.constant_initializer(0.0))
    pop_mean = __get_variable(name_prefix + 'mean', inputs.get_shape()[-1], tf.constant_initializer(0.0),
                              trainable=False)
    pop_var = __get_variable(name_prefix + 'variance', inputs.get_shape()[-1], tf.constant_initializer(1.0),
                             trainable=False)
    axis = list(range(len(inputs.get_shape()) - 1))

    def in_training():
        batch_mean, batch_var = tf.nn.moments(inputs, axis)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

    def in_evaluation():
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

    return tf.cond(is_training, lambda: in_training(), lambda: in_evaluation())


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def prelu(_x, shared=False, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        if shared:
            _alpha = tf.get_variable("prelu", shape=[1], dtype=_x.dtype,
                                     initializer=tf.constant_initializer(0.1))
        else:
            _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1], dtype=_x.dtype,
                                     initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def zrelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha1 = tf.get_variable("alpha1", shape=[1], dtype=_x.dtype, initializer=tf.constant_initializer(1))
        _alpha2 = tf.get_variable("alpha2", shape=[1], dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return _alpha1 * tf.maximum(0.0, _x) + _alpha2 * tf.minimum(0.0, _x)


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
