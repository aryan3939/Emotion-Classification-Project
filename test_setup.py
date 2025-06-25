import os
import numpy as np
import pandas as pd
import librosa
import pickle
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionModelTester:
    """
    A comprehensive testing class for the Mars Emotion Classification model
    """
    
    def __init__(self, model_path='final_emotion_model.h5'):
        """
        Initialize the model tester
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.normalization_params = None
        self.emotion_mapping = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
    def load_model_components(self):
        """Load the trained model and associated components"""
        try:
            # Load the trained model
            self.model = load_model(self.model_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
            
            # Load label encoder
            if os.path.exists('emotion_label_encoder.pkl'):
                self.label_encoder = joblib.load('emotion_label_encoder.pkl')
                print("✓ Label encoder loaded successfully")
            else:
                print("⚠ Label encoder not found, creating default one")
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(list(self.emotion_mapping.values()))
            
            # Load normalization parameters
            if os.path.exists('normalization_params.pkl'):
                with open('normalization_params.pkl', 'rb') as f:
                    self.normalization_params = pickle.load(f)
                print("✓ Normalization parameters loaded successfully")
            else:
                print("⚠ Normalization parameters not found, using defaults")
                self.normalization_params = {'mean': 0.0, 'std': 1.0}
                
        except Exception as e:
            print(f"✗ Error loading model components: {e}")
            return False
        
        return True
    
    def extract_advanced_mel_features(self, file_path, sr=22050, n_mels=77, max_pad_len=174):
        """
        Extract mel-spectrogram features from audio file
        
        Args:
            file_path (str): Path to audio file
            sr (int): Sample rate
            n_mels (int): Number of mel bands
            max_pad_len (int): Maximum padding length
            
        Returns:
            np.array: Extracted features or None if error
        """
        try:
            # Load audio with 3 second duration
            audio, sample_rate = librosa.load(file_path, sr=sr, duration=3.0)
            
            # Advanced preprocessing: Pre-emphasis filter
            pre_emphasis = 0.97
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Pad or truncate to consistent length
            target_length = sr * 3
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract mel-spectrogram with optimized parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, 
                n_mels=n_mels,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window='hann'
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
            
            # Normalize to [-1, 1] range
            mel_spec_normalized = 2 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1
            
            # Pad/truncate to fixed time dimension
            if mel_spec_normalized.shape[1] < max_pad_len:
                pad_width = max_pad_len - mel_spec_normalized.shape[1]
                mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_normalized = mel_spec_normalized[:, :max_pad_len]
            
            return mel_spec_normalized.T  # Shape: (174, 77)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def predict_emotion(self, file_path):
        """
        Predict emotion from audio file
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            tuple: (predicted_emotion, confidence, all_probabilities)
        """
        if self.model is None:
            print("Model not loaded. Please call load_model_components() first.")
            return None, 0.0, {}
        
        try:
            # Extract features
            features = self.extract_advanced_mel_features(file_path)
            
            if features is None:
                return None, 0.0, {}
            
            # Reshape for model input: (174, 77) -> (77, 174, 1)
            features_reshaped = features.T[np.newaxis, ..., np.newaxis]
            
            # Normalize using training statistics
            features_normalized = (features_reshaped - self.normalization_params['mean']) / self.normalization_params['std']
            
            # Make prediction
            predictions = self.model.predict(features_normalized, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_emotion = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Create probability dictionary
            all_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(predictions[0][i])
                for i in range(len(self.label_encoder.classes_))
            }
            
            return predicted_emotion, confidence, all_probabilities
            
        except Exception as e:
            print(f"Error predicting emotion for {file_path}: {e}")
            return None, 0.0, {}
    
    def test_on_dataset(self, test_data_path):
        """
        Test the model on a dataset of audio files
        
        Args:
            test_data_path (str): Path to directory containing test audio files
        """
        if not os.path.exists(test_data_path):
            print(f"Test data path does not exist: {test_data_path}")
            return
        
        predictions = []
        true_labels = []
        file_names = []
        
        print(f"Testing model on files in: {test_data_path}")
        
        # Process all wav files in the directory
        for file_name in os.listdir(test_data_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(test_data_path, file_name)
                
                # Extract true emotion from filename (RAVDESS format)
                try:
                    emotion_code = file_name.split('-')[2]
                    true_emotion = self.emotion_mapping.get(emotion_code, 'unknown')
                    
                    # Predict emotion
                    predicted_emotion, confidence, _ = self.predict_emotion(file_path)
                    
                    if predicted_emotion is not None:
                        predictions.append(predicted_emotion)
                        true_labels.append(true_emotion)
                        file_names.append(file_name)
                        
                        print(f"File: {file_name[:30]:<30} | True: {true_emotion:<10} | Predicted: {predicted_emotion:<10} | Confidence: {confidence:.2%}")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        
        # Calculate metrics
        if predictions and true_labels:
            accuracy = accuracy_score(true_labels, predictions)
            print(f"\n{'='*60}")
            print(f"TESTING RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total files tested: {len(predictions)}")
            print(f"Overall Accuracy: {accuracy:.2%}")
            
            # Classification report
            print(f"\nDetailed Classification Report:")
            print(classification_report(true_labels, predictions))
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions, labels=self.label_encoder.classes_)
            
            # Visualize confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix - Model Testing Results')
            plt.xlabel('Predicted Emotion')
            plt.ylabel('True Emotion')
            plt.tight_layout()
            plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return accuracy, predictions, true_labels
        else:
            print("No valid predictions made.")
            return 0.0, [], []
    
    def test_single_file(self, file_path):
        """
        Test the model on a single audio file
        
        Args:
            file_path (str): Path to audio file
        """
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return
        
        print(f"Testing single file: {file_path}")
        
        predicted_emotion, confidence, all_probabilities = self.predict_emotion(file_path)
        
        if predicted_emotion is not None:
            print(f"\n{'='*50}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*50}")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Predicted Emotion: {predicted_emotion}")
            print(f"Confidence: {confidence:.2%}")
            
            print(f"\nAll Emotion Probabilities:")
            for emotion, prob in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion:>10}: {prob:.2%}")
        else:
            print("Failed to make prediction.")

def main():
    """Main testing function"""
    print("="*60)
    print("MARS EMOTION CLASSIFICATION - MODEL TESTING")
    print("="*60)
    
    # Initialize tester
    tester = EmotionModelTester()
    
    # Load model components
    if not tester.load_model_components():
        print("Failed to load model components. Exiting.")
        return
    
    # Example usage - you can modify these paths
    print(f"\nModel loaded successfully!")
    print(f"Supported emotions: {list(tester.label_encoder.classes_)}")
    
    # Test options
    print(f"\nTesting Options:")
    print(f"1. Test single file: tester.test_single_file('path/to/audio.wav')")
    print(f"2. Test dataset: tester.test_on_dataset('path/to/test/directory')")
    
    # Example single file test (uncomment and modify path as needed)
    # tester.test_single_file('sample_audio.wav')
    
    # Example dataset test (uncomment and modify path as needed)
    # tester.test_on_dataset('test_audio_files/')
    
    print(f"\nTo use this script:")
    print(f"1. Ensure your model file 'final_emotion_model.h5' is in the current directory")
    print(f"2. Place test audio files in a directory")
    print(f"3. Call tester.test_on_dataset('your_test_directory')")
    
if __name__ == "__main__":
    main()
