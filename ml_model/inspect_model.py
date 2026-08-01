import tensorflow as tf
import joblib
import image_encoder

model = tf.keras.models.load_model(
    "results/maize/maize_best_model.keras"
)

print("\nTop Level Layers\n")

for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer))

print("\nImageEncoder Sub Layers\n")

encoder = model.get_layer("ImageEncoder")

if hasattr(encoder, "layers"):
    for i, layer in enumerate(encoder.layers):
        print(i, layer.name, type(layer))
else:
    print("ImageEncoder has no .layers attribute")