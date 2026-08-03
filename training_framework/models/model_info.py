import tensorflow as tf


def print_model_statistics(model):

    trainable = int(
        sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    )

    non_trainable = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )

    total = trainable + non_trainable

    print("\n")
    print("=" * 60)
    print("Model Statistics")
    print("=" * 60)

    print(f"Total Parameters       : {total:,}")
    print(f"Trainable Parameters   : {trainable:,}")
    print(f"Frozen Parameters      : {non_trainable:,}")

    print("=" * 60)