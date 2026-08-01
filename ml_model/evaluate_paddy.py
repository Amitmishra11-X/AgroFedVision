import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

MODEL_PATH = "outputs/checkpoints/best_paddy_model.keras"

DATASET_PATH = r"D:\Download1\archive\paddy-disease-classification"

TRAIN_FOLDER = os.path.join(DATASET_PATH,"train_images")

TRAIN_CSV = os.path.join(DATASET_PATH,"train.csv")

model = load_model(MODEL_PATH)

print("Model Loaded Successfully")
from sklearn.model_selection import train_test_split

SEED = 42

df = pd.read_csv(TRAIN_CSV)

classes = sorted(df.label.unique())

class_to_idx = {
    c:i
    for i,c in enumerate(classes)
}

idx_to_class = {
    i:c
    for c,i in class_to_idx.items()
}

df["label_idx"] = df.label.map(class_to_idx)

train_df,val_df = train_test_split(

    df,

    test_size=0.20,

    random_state=SEED,

    stratify=df.label

)
def load_image(path,label):

    image=tf.io.read_file(path)

    image=tf.image.decode_jpeg(image,channels=3)

    image=tf.image.resize(image,IMAGE_SIZE)

    image=preprocess_input(image)

    return image,label


paths=[]

labels=[]

for _,row in val_df.iterrows():

    paths.append(

        os.path.join(

            TRAIN_FOLDER,

            row.label,

            row.image_id

        )

    )

    labels.append(row.label_idx)

dataset=tf.data.Dataset.from_tensor_slices(

    (paths,labels)

)

dataset=dataset.map(load_image)

dataset=dataset.batch(BATCH_SIZE)
pred=model.predict(dataset)

pred=np.argmax(pred,axis=1)

true=np.array(labels)
acc=accuracy_score(true,pred)

precision=precision_score(

    true,

    pred,

    average="weighted"

)

recall=recall_score(

    true,

    pred,

    average="weighted"

)

f1=f1_score(

    true,

    pred,

    average="weighted"

)

print()

print("Accuracy :",acc)

print("Precision :",precision)

print("Recall :",recall)

print("F1 :",f1)
report=classification_report(

    true,

    pred,

    target_names=classes

)

print(report)

with open(

    "outputs/reports/classification_report.txt",

    "w"

) as f:

    f.write(report)
cm=confusion_matrix(

    true,

    pred

)

plt.figure(figsize=(10,10))

plt.imshow(cm)

plt.colorbar()

plt.xticks(

    np.arange(len(classes)),

    classes,

    rotation=90

)

plt.yticks(

    np.arange(len(classes)),

    classes

)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.savefig(

    "outputs/figures/confusion_matrix.png",

    dpi=300

)

plt.show()
predictions=pd.DataFrame({

    "True Label":[idx_to_class[i] for i in true],

    "Predicted":[idx_to_class[i] for i in pred]

})

predictions.to_csv(

    "outputs/predictions/predictions.csv",

    index=False

)

print("Evaluation Completed")