import os
import joblib
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# IMPORTANT: These labels must exactly match the labels used during training
LABELS = [
    "Healthy",
    "High_Risk"
]

encoder.fit(LABELS)

os.makedirs("results/maize", exist_ok=True)

joblib.dump(
    encoder,
    "results/maize/label_encoder.pkl"
)

print("Maize encoder saved successfully!")
print("Classes:", encoder.classes_)