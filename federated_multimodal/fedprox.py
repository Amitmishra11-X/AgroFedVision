import tensorflow as tf

MU = 0.01

def fedprox_loss(
        y_true,
        y_pred,
        model,
        global_weights):

    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true,
        y_pred
    )

    prox_term = 0

    for w, gw in zip(
            model.trainable_weights,
            global_weights):

        prox_term += tf.reduce_sum(
            tf.square(w - gw)
        )

    return tf.reduce_mean(
        ce_loss +
        (MU / 2.0) * prox_term
    )