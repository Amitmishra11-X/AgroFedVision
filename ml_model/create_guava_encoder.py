import os
import joblib
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit([
    "Healthy",
    "High_Risk",
    "Moderate_Risk"
])

os.makedirs("results/phase1_guava", exist_ok=True)

joblib.dump(
    encoder,
    "results/phase1_guava/label_encoder.pkl"
)

print("Encoder saved successfully!")
print("Classes:", encoder.classes_)