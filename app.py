from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("brain_tumor_model.h5")

# Classes from Kaggle dataset
class_names = [
    "pituitary",
    "notumor",
    "meningioma",
    "glioma"
]

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template(
            "index.html",
            error="Please select an MRI image."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an MRI image."
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    try:
        # Load MRI image
        img = image.load_img(
            filepath,
            target_size=(224, 224)
        )

        # Convert image to array
        img_array = image.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize image
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction[0])

        predicted_class = class_names[predicted_index]

        confidence = float(
            prediction[0][predicted_index] * 100
        )

        # Medical assistant information
        if predicted_class == "notumor":
            advice = """
            The model classified this MRI image as No Tumor.

            This is an AI-based prediction and not a medical diagnosis.
            If you have symptoms or concerns, please consult a qualified
            healthcare professional.
            """
        else:
            advice = f"""
            The model classified this MRI image as {predicted_class}.

            This is an AI-based prediction and not a medical diagnosis.
            Please consult a qualified healthcare professional for
            proper examination and confirmation.
            """

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            advice=advice,
            image_path="/" + filepath
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Error processing image: " + str(e)
        )


if __name__ == "__main__":
    print("Open this link in your browser: http://127.0.0.1:5000")
    app.run(debug=True)