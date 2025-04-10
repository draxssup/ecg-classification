from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.path.join("model", "saved_model")

# Load the model once at the start
loaded_model = tf.saved_model.load(MODEL_PATH)
inference_func = loaded_model.signatures["serving_default"]

# Health Check Route to keep the app alive
@app.route("/health_check", methods=["GET", "HEAD"])
def health_check():
    return jsonify({"status": "ECG model server is live"}), 200

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    print("Received request to /predict")
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        # Resize and preprocess the image
        img = Image.open(filepath).convert("RGB").resize((256, 182))  # Resize to model input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 182, 256, 3)

        # Convert to tensor and make prediction
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        predictions = inference_func(input_6=input_tensor)

        # Get predicted class
        predicted_class = np.argmax(list(predictions.values())[0].numpy())
        print("Predicted class:", predicted_class)

        # Map prediction to class name
        class_names = [
            "Normal",
            "Myocardial Infarction",
            "History of MI",
            "Abnormal Heartbeat",
        ]
        result = (
            class_names[predicted_class]
            if predicted_class < len(class_names)
            else "Unknown"
        )

        # Return prediction and recommendation
        recommendation = {
            "Normal": "Your ECG appears normal. Maintain a healthy lifestyle.",
            "Myocardial Infarction": "Signs of heart attack detected. Seek emergency care immediately.",
            "History of MI": "Signs of past heart attack. Follow up with a cardiologist.",
            "Abnormal Heartbeat": "Irregular heartbeat detected. Please consult a doctor.",
        }

        return jsonify({"prediction": result, "recommendation": recommendation[result]})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    # Make sure the port is dynamically set for Render deployment
    port = int(os.environ.get("PORT", 5000))  # Get port dynamically for deployment
    # Run the app with threaded=True for handling multiple requests and use_reloader=False to avoid auto-reload on Render
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
