"""
Batch Image Prediction - Test multiple images at once
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# Load model
try:
    model = tf.keras.models.load_model('intel_image_classifier.h5')
    print("Model loaded successfully!")
except:
    print("ERROR: Model not found! Run image_classification_project.py first.")
    exit()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_batch_images(folder_path):
    """Predict all images in a folder"""
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print("No image files found in the folder!")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    results = []
    
    # Process each image
    for i, img_path in enumerate(image_files):
        try:
            # Load and preprocess
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            
            results.append({
                'file': os.path.basename(img_path),
                'class': predicted_class,
                'confidence': confidence,
                'all_probs': predictions[0]
            })
            
            print(f"[{i+1}/{len(image_files)}] {os.path.basename(img_path)}: {predicted_class} ({confidence*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Display summary
    print("\n" + "="*70)
    print("BATCH PREDICTION RESULTS")
    print("="*70)
    
    for result in results:
        print(f"{result['file']:<30} -> {result['class']:<10} ({result['confidence']*100:.1f}%)")
    
    # Show class distribution
    class_counts = {}
    for result in results:
        class_name = result['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nClass Distribution:")
    print("-" * 30)
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        print(f"{class_name}: {count} images")
    
    return results

if __name__ == "__main__":
    print("Batch Image Prediction Tool")
    print("Process all images in a folder")
    print("-" * 40)
    
    folder = input("Enter folder path containing images: ").strip().strip('"')
    
    if os.path.exists(folder):
        predict_batch_images(folder)
    else:
        print("Folder not found!")