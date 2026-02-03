try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    print("Flask is not installed. Please install it with: pip install flask")
    exit(1)

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Get absolute path to model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'intel_image_classifier.h5')

if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'!")
    print("Please ensure the model is trained and saved.")
    exit(1)

# Load the trained model
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
    
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(image):
    try:
        # Preprocess image
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Image array shape: {img_array.shape}")
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return predicted_class, confidence, predictions[0]
        
    except Exception as e:
        print(f"Error in predict_image: {e}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        print(f"Processing file: {file.filename}")
        
        # Read and process image
        image = Image.open(file.stream)
        print(f"Image mode: {image.mode}, Size: {image.size}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted image to RGB")
        
        # Get prediction
        predicted_class, confidence, all_predictions = predict_image(image)
        print(f"Prediction: {predicted_class} with confidence: {confidence:.4f}")
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Prepare all class probabilities
        class_probabilities = {class_names[i]: float(all_predictions[i]) for i in range(len(class_names))}
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'accuracy_percentage': f"{confidence * 100:.2f}%",
            'image': img_str,
            'all_predictions': class_probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=False, host='127.0.0.1', port=5000)