# Fire Detection Using Deep Learning

A simple internship-project implementation using Python, TensorFlow/Keras, Flask, HTML and CSS.

## Project structure
- `app.py` — Flask web application and model prediction
- `train.py` — CNN training script
- `templates/index.html` — frontend
- `static/style.css` — styling
- `dataset/fire/` — put fire images here
- `dataset/no_fire/` — put non-fire images here
- `model/` — trained model is saved here

## Setup
1. Install Python 3.10 or 3.11.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   `pip install -r requirements.txt`
4. Add a sufficiently large and diverse dataset to:
   - `dataset/fire/`
   - `dataset/no_fire/`
5. Train the model:
   `python train.py`
6. Start the website:
   `python app.py`
7. Open `http://127.0.0.1:5000` in a browser.

## Important
The website does **not** use random predictions. It requires a trained TensorFlow/Keras model. For a meaningful accuracy result, train it with a suitable fire/non-fire dataset and evaluate it on a separate test set.
