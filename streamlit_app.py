import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Brush Script MT', cursive;
        font-size: 3rem;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = 'intel_image_classifier.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            st.error("Model file not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction function
def predict_image(image, model):
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    return predicted_class, confidence, predictions[0], class_names

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Intel Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload any image to classify it into: Buildings, Forest, Glacier, Mountain, Sea, or Street</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to classify"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner('Analyzing image...'):
            predicted_class, confidence, all_predictions, class_names = predict_image(image, model)
        
        # Display results
        st.success(f"**Predicted Class:** {predicted_class.upper()}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
        
        # Show all predictions
        st.subheader("All Class Probabilities:")
        for i, class_name in enumerate(class_names):
            probability = all_predictions[i] * 100
            st.progress(probability/100, text=f"{class_name}: {probability:.2f}%")

if __name__ == "__main__":
    main()