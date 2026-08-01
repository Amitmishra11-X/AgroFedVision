import os
import random
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from image_encoder import build_image_encoder

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 5

LR_STAGE1 = 3e-4
LR_STAGE2 = 1e-5

DATASET_DIR = r"D:\Download1\archive (1)\Corn Disease detection"

RESULTS_DIR = "results/maize"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("AgroFedVision - Phase 1 (Maize)")
print("=" * 60)
print("TensorFlow :", tf.__version__)
def load_dataset():

    image_paths = []
    labels = []

    classes = {
        "Healthy corn": "Healthy",
        "Infected": "High_Risk"
    }

    for folder_name, label in classes.items():

        folder = os.path.join(DATASET_DIR, folder_name)

        if not os.path.exists(folder):
            raise FileNotFoundError(folder)

        for root, _, files in os.walk(folder):

            for file in files:

                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):

                    image_paths.append(
                        os.path.join(root, file)
                    )

                    labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    print("\nImages :", len(image_paths))

    unique, counts = np.unique(labels, return_counts=True)

    print("\nClass Distribution")

    for u, c in zip(unique, counts):
        print(f"{u:15s} {c}")

    return image_paths, labels
def encode_labels(labels):

    encoder = LabelEncoder()

    y = encoder.fit_transform(labels)

    print("\nClasses")

    for i, cls in enumerate(encoder.classes_):
        print(i, "->", cls)

    return y, encoder
def split_dataset(paths, labels):

    X_train, X_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED
    )

    print("\nDataset Split")

    print("Train :", len(X_train))
    print("Validation :", len(X_val))
    print("Test :", len(X_test))

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = tf.cast(image, tf.float32)

    return image, label
augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.10),

    tf.keras.layers.RandomZoom(0.10),

    tf.keras.layers.RandomContrast(0.10)

])
def create_dataset(paths, labels, training=False):

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:

        ds = ds.map(
            lambda x, y: (augmentation(x), y),
            num_parallel_calls=AUTOTUNE
        )

        ds = ds.shuffle(
            1000,
            seed=SEED
        )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds
def get_class_weights(labels):

    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    weights = dict(
        zip(classes, weights)
    )

    print("\nClass Weights")

    print(weights)

    return weights
# ============================================================
# BUILD MODEL
# ============================================================

def build_model(num_classes):

    encoder = build_image_encoder(
        embed_dim=256,
        dropout=0.30,
        trainable_backbone=False
    )

    inputs = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="input_image"
    )

    x = encoder(inputs)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.40)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="Maize_Classifier"
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LR_STAGE1
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("\nBuilding Model...\n")

    model.summary()

    return model
# ============================================================
# CALLBACKS
# ============================================================

def build_callbacks():

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=os.path.join(
                RESULTS_DIR,
                "maize_best_model.keras"
            ),

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=2,

            verbose=1

        ),

        tf.keras.callbacks.CSVLogger(

            os.path.join(
                RESULTS_DIR,
                "training_log.csv"
            )

        )

    ]

    return callbacks
# ============================================================
# STAGE 1
# ============================================================

def train_stage1(

    model,

    train_ds,

    val_ds,

    class_weights

):

    print("\n" + "="*60)
    print("Stage 1 Training")
    print("="*60)

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE1,

        class_weight=class_weights,

        callbacks=build_callbacks(),

        verbose=1

    )

    return history
# ============================================================
# UNFREEZE
# ============================================================

def unfreeze_backbone(model):

    print("\nFine-tuning Block6 + Block7")

    encoder = model.get_layer("ImageEncoder")

    encoder.trainable = True

    trainable = False

    for layer in encoder.layers:

        if layer.name.startswith("block6"):

            trainable = True

        if layer.name.startswith("block7"):

            trainable = True

        layer.trainable = trainable
# ============================================================
# STAGE 2
# ============================================================

def train_stage2(

    model,

    train_ds,

    val_ds,

    class_weights

):

    print("\n" + "="*60)
    print("Stage 2 Fine Tuning")
    print("="*60)

    unfreeze_backbone(model)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LR_STAGE2

        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE2,

        class_weight=class_weights,

        callbacks=build_callbacks(),

        verbose=1

    )

    return history
# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(

    model,

    train_ds,

    val_ds,

    class_weights

):

    history1 = train_stage1(

        model,

        train_ds,

        val_ds,

        class_weights

    )

    history2 = train_stage2(

        model,

        train_ds,

        val_ds,

        class_weights

    )

    return history1, history2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_ds, encoder):

    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    predictions = model.predict(test_ds, verbose=1)

    y_pred = np.argmax(predictions, axis=1)

    y_true = np.concatenate(
        [y.numpy() for _, y in test_ds],
        axis=0
    )

    accuracy = np.mean(y_true == y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )

    print("\nClassification Report\n")
    print(report)

    with open(
        os.path.join(
            RESULTS_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as f:

        f.write(report)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    plt.title("Confusion Matrix")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return accuracy, precision, recall, f1
# ============================================================
# TRAINING CURVES
# ============================================================

def plot_history(history1, history2):

    acc = (
        history1.history["accuracy"] +
        history2.history["accuracy"]
    )

    val_acc = (
        history1.history["val_accuracy"] +
        history2.history["val_accuracy"]
    )

    loss = (
        history1.history["loss"] +
        history2.history["loss"]
    )

    val_loss = (
        history1.history["val_loss"] +
        history2.history["val_loss"]
    )

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        acc,
        label="Train"
    )

    plt.plot(
        epochs,
        val_acc,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        loss,
        label="Train"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "loss.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# ============================================================
# MAIN
# ============================================================

def main():

    paths, labels = load_dataset()

    labels, encoder = encode_labels(labels)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test

    ) = split_dataset(
        paths,
        labels
    )

    train_ds = create_dataset(
        X_train,
        y_train,
        training=True
    )

    val_ds = create_dataset(
        X_val,
        y_val
    )

    test_ds = create_dataset(
        X_test,
        y_test
    )

    class_weights = get_class_weights(
        y_train
    )

    model = build_model(
        len(encoder.classes_)
    )

    history1, history2 = train_model(

        model,

        train_ds,

        val_ds,

        class_weights

    )

    model.load_weights(

        os.path.join(

            RESULTS_DIR,

            "maize_best_model.keras"

        )

    )

    evaluate_model(

        model,

        test_ds,

        encoder

    )

    plot_history(

        history1,

        history2

    )

    print("\nTraining Completed Successfully.")

    print("Results Saved In :", RESULTS_DIR)
# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()