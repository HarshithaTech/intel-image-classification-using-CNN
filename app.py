from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('intel_image_classifier.h5')
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(image):
    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    
    return predicted_class, confidence, predictions[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get prediction
        predicted_class, confidence, all_predictions = predict_image(image)
        
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
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)