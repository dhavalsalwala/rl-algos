import tensorflow as tf
from keras import backend as K

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)
