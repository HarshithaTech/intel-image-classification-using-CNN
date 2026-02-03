# Intel Image Classification Project using Deep Learning
# Dataset: 6 classes - buildings, forest, glacier, mountain, sea, street

# ============================================================================
# IMPORTS
# ============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

# Set dataset paths (adjust these paths according to your dataset location)
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test', 'seg_test')

# Check if directories exist
if not os.path.exists(train_dir):
    print(f"Error: Training directory not found at {train_dir}")
    print("Please ensure the dataset is properly extracted.")
    exit(1)
    
if not os.path.exists(test_dir):
    print(f"Error: Test directory not found at {test_dir}")
    print("Please ensure the dataset is properly extracted.")
    exit(1)

print(f"Training directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
NUM_CLASSES = 6

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 20% for validation
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
try:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
except Exception as e:
    print(f"Error loading training data: {e}")
    print(f"Please check if the directory {train_dir} exists and contains subdirectories with images.")
    exit(1)

# Load validation data
try:
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
except Exception as e:
    print(f"Error loading validation data: {e}")
    exit(1)

# Load test data
try:
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
except Exception as e:
    print(f"Error loading test data: {e}")
    print(f"Please check if the directory {test_dir} exists and contains subdirectories with images.")
    exit(1)

print("Class indices:", train_generator.class_indices)

# ============================================================================
# MODEL BUILDING
# ============================================================================

model = keras.Sequential([
    # First Conv Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Second Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Third Conv Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Fourth Conv Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# ============================================================================
# TRAINING
# ============================================================================

EPOCHS = 20

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1
)

# ============================================================================
# EVALUATION
# ============================================================================

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ============================================================================
# SAVING MODEL
# ============================================================================

model.save('intel_image_classifier.h5')
print("\nModel saved as 'intel_image_classifier.h5'")

# ============================================================================
# SINGLE IMAGE PREDICTION
# ============================================================================

def predict_single_image(image_path, model):
    """
    Predict the class of a single image
    """
    # Class names
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None, None
            
        # Load and preprocess the image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        
        # Display results
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Predicted: {predicted_class} (Confidence: {confidence:.2f})')
        plt.axis('off')
        plt.show()
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

# Example usage (uncomment and provide a valid image path):
# predict_single_image('path_to_your_test_image.jpg', model)

print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)
print("✓ Model trained and evaluated")
print("✓ Training plots generated")
print("✓ Model saved as 'intel_image_classifier.h5'")
print("✓ Single image prediction function ready")
print("\nTo predict a new image, use:")
print("predict_single_image('your_image_path.jpg', model)")