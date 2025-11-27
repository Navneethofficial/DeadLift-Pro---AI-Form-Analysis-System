"""
Script to train the form classifier using labeled images
Place good form images in 'good/' folder and bad form images in 'bad/' folder
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from pose_analyzer import PoseAnalyzer

class ModelTrainer:
    def __init__(self, good_folder='good', bad_folder='bad'):
        self.good_folder = good_folder
        self.bad_folder = bad_folder
        self.analyzer = PoseAnalyzer()
        
    def load_training_data(self):
        """Load and process training images"""
        x_train_good = []
        x_train_bad = []
        
        print("Loading good posture images...")
        if os.path.exists(self.good_folder):
            for filename in os.listdir(self.good_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(self.good_folder, filename)
                    features = self.extract_features(filepath)
                    if features is not None:
                        x_train_good.append(features)
                        print(f"  ✓ Processed {filename}")
        
        print(f"\nLoading bad posture images...")
        if os.path.exists(self.bad_folder):
            for filename in os.listdir(self.bad_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(self.bad_folder, filename)
                    features = self.extract_features(filepath)
                    if features is not None:
                        x_train_bad.append(features)
                        print(f"  ✓ Processed {filename}")
        
        print(f"\n📊 Training Data Summary:")
        print(f"   Good postures: {len(x_train_good)}")
        print(f"   Bad postures: {len(x_train_bad)}")
        
        return x_train_good, x_train_bad
    
    def extract_features(self, image_path):
        """Extract pose features from an image"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"  ✗ Could not read {image_path}")
                return None
            
            # Detect keypoints
            points = self.analyzer.detect_keypoints(frame)
            
            # Convert to feature vector
            features = []
            for point in points:
                if point:
                    features.append(point[0])
                    features.append(point[1])
                else:
                    features.append(-1)
                    features.append(-1)
            
            # Add computed features (angles, distances)
            metrics = self.analyzer.calculate_metrics(points)
            
            # Add angle features if available
            if 'back_angle' in metrics:
                features.append(metrics['back_angle'])
            else:
                features.append(-1)
            
            if 'hip_angle' in metrics:
                features.append(metrics['hip_angle'])
            else:
                features.append(-1)
            
            if 'knee_angle' in metrics:
                features.append(metrics['knee_angle'])
            else:
                features.append(-1)
            
            return features
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {str(e)}")
            return None
    
    def train_model(self, model_type='ridge'):
        """Train the classifier"""
        
        # Load data
        x_train_good, x_train_bad = self.load_training_data()
        
        if len(x_train_good) == 0 or len(x_train_bad) == 0:
            print("\n❌ Error: Need at least one image in both 'good' and 'bad' folders")
            return None
        
        # Combine data
        X = x_train_good + x_train_bad
        y = [1] * len(x_train_good) + [0] * len(x_train_bad)
        
        # Choose model
        print(f"\n🤖 Training {model_type} model...")
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            print(f"Unknown model type: {model_type}")
            return None
        
        # Train model
        model.fit(X, y)
        
        # Evaluate with cross-validation
        if len(X) >= 5:  # Need at least 5 samples for CV
            scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='accuracy')
            print(f"\n📈 Cross-validation scores: {scores}")
            print(f"   Average accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
        
        # Save model
        model_filename = f'trained_model_{model_type}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✅ Model saved as '{model_filename}'")
        
        return model
    
    def test_model(self, model, test_image_path):
        """Test the model on a single image"""
        print(f"\n🧪 Testing model on {test_image_path}...")
        
        features = self.extract_features(test_image_path)
        if features is None:
            print("Could not extract features from test image")
            return
        
        prediction = model.predict([features])[0]
        
        print(f"   Raw prediction: {prediction:.4f}")
        print(f"   Classification: {'Good Form' if prediction > 0.5 else 'Bad Form'}")
        
        # Also show the actual analysis
        result = self.analyzer.analyze_image(test_image_path)
        print(f"   Form score: {result['score']*100:.1f}%")


def main():
    """Main training function"""
    print("=" * 60)
    print("DeadLift Pro - Model Training")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    # Train different models
    models = {
        'ridge': None,
        'logistic': None,
        'random_forest': None
    }
    
    print("\nTraining multiple models for comparison...\n")
    
    for model_type in models.keys():
        print(f"\n{'='*60}")
        models[model_type] = trainer.train_model(model_type)
        print(f"{'='*60}")
    
    # Test on a sample image if available
    test_image = None
    if os.path.exists('good') and os.listdir('good'):
        test_image = os.path.join('good', os.listdir('good')[0])
    elif os.path.exists('bad') and os.listdir('bad'):
        test_image = os.path.join('bad', os.listdir('bad')[0])
    
    if test_image and models['ridge']:
        trainer.test_model(models['ridge'], test_image)
    
    print("\n" + "="*60)
    print("Training complete! You can now use the trained models.")
    print("="*60)
    print("\nTo use a trained model in the app:")
    print("1. Update pose_analyzer.py to load the .pkl file")
    print("2. Replace the default classifier initialization")
    print("3. Restart the Flask application")


if __name__ == '__main__':
    # Check if training folders exist
    if not os.path.exists('good'):
        os.makedirs('good')
        print("Created 'good' folder - add your good form images here")
    
    if not os.path.exists('bad'):
        os.makedirs('bad')
        print("Created 'bad' folder - add your bad form images here")
    
    # Check if there are images to train on
    good_count = len([f for f in os.listdir('good') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('good') else 0
    bad_count = len([f for f in os.listdir('bad') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists('bad') else 0
    
    if good_count == 0 or bad_count == 0:
        print("\n⚠ Warning: No training images found!")
        print("Please add images to 'good' and 'bad' folders before training.")
        print(f"Current counts - Good: {good_count}, Bad: {bad_count}")
    else:
        main()