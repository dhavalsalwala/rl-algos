import tensorflow as tf


def clip_grads(grads, clip_rate):
    if clip_rate is None:
        return grads
    for i, (grad, var) in enumerate(grads):
        if grad is not None:
            grads[i] = (tf.clip_by_norm(grad, clip_rate), var)
    return grads