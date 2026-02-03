"""
Quick Web App Test - Intel Image Classifier
Run this to start the web interface for image prediction
"""

import os
import sys

def check_model_exists():
    if not os.path.exists('intel_image_classifier.h5'):
        print("❌ Model file 'intel_image_classifier.h5' not found!")
        print("Please run 'image_classification_project.py' first to train and save the model.")
        return False
    return True

def start_webapp():
    if check_model_exists():
        print("🚀 Starting Intel Image Classifier Web App...")
        print("📱 Open your browser and go to: http://127.0.0.1:5000")
        print("🖼️ Upload any image to get instant predictions!")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            from app import app
            app.run(debug=True, host='127.0.0.1', port=5000)
        except ImportError:
            print("❌ Flask not installed. Run: pip install flask")
        except Exception as e:
            print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    start_webapp()