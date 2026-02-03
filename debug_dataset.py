#!/usr/bin/env python3
"""
Debug script to check dataset structure and image loading
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get current directory
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')

print("=== Dataset Debug Information ===")
print(f"Base directory: {base_dir}")
print(f"Training directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Check if directories exist
print(f"\nTraining directory exists: {os.path.exists(train_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

# List subdirectories
if os.path.exists(train_dir):
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print(f"Training classes found: {train_classes}")
    
    # Count images in each class
    for class_name in train_classes:
        class_path = os.path.join(train_dir, class_name)
        image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {class_name}: {image_count} images")

if os.path.exists(test_dir):
    test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    print(f"Test classes found: {test_classes}")

# Test data generator
print("\n=== Testing Data Generator ===")
try:
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset=None
    )
    print("SUCCESS: Data generator created successfully")
    print(f"Classes found: {test_generator.class_indices}")
    print(f"Total samples: {test_generator.samples}")
except Exception as e:
    print(f"ERROR: Error creating data generator: {e}")

print("\n=== Debug Complete ===")