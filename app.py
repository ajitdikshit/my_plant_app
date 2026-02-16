import os
import io
import numpy as np
import tensorflow as tf
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from rembg import remove

app = Flask(__name__)
CORS(app)  # Allows GitHub Pages to talk to this API

# 1. MODEL DOWNLOAD LOGIC
# Replace 'YOUR_FILE_ID' with the ID from your Google Drive link
FILE_ID = '1kCaUvVNhsnDx5ptQEJ9-NMSWiz81qePb' 
model_path = 'master_plant_model.h5'

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    gdown.download(url, model_path, quiet=False)

# 2. LOAD MODEL (with Keras 3 compatibility fix)
# compile=False avoids the 'Value Error: Invalid dtype' issue
model = tf.keras.models.load_model(model_path, compile=False)

# Your 38 class names in alphabetical order
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    img = Image.open(file.stream)

    # 3. BACKGROUND REMOVAL (Clean the image for better AI accuracy)
    # Using alpha_matting to prevent the AI from deleting the leaf edges
    no_bg = remove(img, alpha_matting=True, alpha_matting_foreground_threshold=240)
    white_bg = Image.new("RGB", no_bg.size, (255, 255, 255))
    if no_bg.mode == 'RGBA':
        white_bg.paste(no_bg, mask=no_bg.split()[3])
    else:
        white_bg = no_bg.convert("RGB")
    
    # 4. PRE-PROCESSING
    img_resized = white_bg.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # Normalize if your model expects it
    img_array = np.expand_dims(img_array, axis=0)

    # 5. PREDICTION
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(score)
    
    # Clean up the name for the user (replace underscores with spaces)
    display_name = class_names[class_idx].replace("___", " - ").replace("_", " ")

    return jsonify({
        "prediction": display_name,
        "confidence": round(float(np.max(score)) * 100, 2)
    })

# 6. RENDER PORT BINDING (The final cure for 'No open ports detected')
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)