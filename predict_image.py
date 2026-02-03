"""
Simple Image Prediction Tool
Upload any image and get instant classification results
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, Tk

# Load model
try:
    model = tf.keras.models.load_model('intel_image_classifier.h5')
    print("Model loaded successfully!")
except:
    print("ERROR: Model file 'intel_image_classifier.h5' not found!")
    print("Please run 'image_classification_project.py' first to train the model.")
    exit()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(image_path):
    """Predict image class with confidence scores"""
    
    # Load and preprocess image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Original Image\nPredicted: {predicted_class.upper()}\nConfidence: {confidence*100:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Show prediction probabilities
    plt.subplot(1, 2, 2)
    probabilities = predictions[0] * 100
    colors = ['red' if i == predicted_class_index else 'lightblue' for i in range(len(class_names))]
    bars = plt.barh(class_names, probabilities, color=colors)
    plt.xlabel('Confidence (%)')
    plt.title('All Class Probabilities', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        plt.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    print("-" * 30)
    
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        marker = ">>> " if i == predicted_class_index else "    "
        print(f"{marker}{class_name}: {prob*100:.2f}%")
    
    return predicted_class, confidence

def select_and_predict():
    """Open file dialog to select image and predict"""
    
    # Hide tkinter root window
    root = Tk()
    root.withdraw()
    
    # Open file dialog
    print("Select an image file to classify...")
    file_path = filedialog.askopenfilename(
        title="Select Image for Classification",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        print(f"Selected: {file_path}")
        predict_image(file_path)
    else:
        print("No file selected.")

if __name__ == "__main__":
    print("Intel Image Classifier - Manual Prediction Tool")
    print("Classes: buildings, forest, glacier, mountain, sea, street")
    print("-" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Select image file")
        print("2. Enter image path manually")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            select_and_predict()
        
        elif choice == '2':
            path = input("Enter image path: ").strip().strip('"')
            if os.path.exists(path):
                predict_image(path)
            else:
                print("File not found!")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")