from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from pathlib import Path
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "fire-detection-demo-key"
BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "static" / "uploads"
MODEL_PATH = BASE / "model" / "fire_detector.keras"
UPLOADS.mkdir(parents=True, exist_ok=True)

model = None
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)

ALLOWED = {"png", "jpg", "jpeg", "webp"}

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED

def predict_image(path):
    if model is None:
        raise RuntimeError("Trained model not found. Add dataset images and run train.py first.")
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    pred = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    # Model labels: 0 = no_fire, 1 = fire
    if pred >= 0.5:
        return "Fire Detected", pred, "danger"
    return "No Fire", 1.0 - pred, "safe"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Please select an image."
        elif not allowed_file(file.filename):
            error = "Supported formats: JPG, JPEG, PNG, WEBP."
        else:
            filename = secure_filename(file.filename)
            save_path = UPLOADS / filename
            file.save(save_path)
            image_url = url_for("static", filename=f"uploads/{filename}")
            try:
                label, confidence, css = predict_image(save_path)
                result = {"label": label, "confidence": round(confidence * 100, 2), "css": css}
            except Exception as exc:
                error = str(exc)
    return render_template("index.html", result=result, image_url=image_url, error=error, model_ready=model is not None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
