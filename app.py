from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# ------------------------
# Flask App Setup
# ------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists (RUNS ONCE)
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------
# Load Model (ONCE)
# ------------------------
model = MobileNetV2(weights="imagenet")
print("✅ MobileNetV2 model loaded")

# ------------------------
# Home Page
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------
# Prediction Route
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template("index.html", error="No image selected")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Image preprocessing
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))   # ✅ REQUIRED SIZE

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]

    label = decoded[1]
    confidence = decoded[2] * 100

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence, 2),
        image_path=filepath
    )

# ------------------------
# Run App
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
