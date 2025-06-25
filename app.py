# app.py - Simple Streamlit App for Mars Emotion Classification

import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Emotion Classification",
    page_icon="🎭",
    layout="centered"
)

class EmotionClassifier:
    """Simple emotion classification class"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
    def load_model_components(self):
        """Load model and components"""
        try:
            # Load the trained model
            if os.path.exists('final_reference_model.h5'):
                self.model = load_model('final_reference_model.h5')
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Model file not found!")
                return False
            
            # Load label encoder if available
            if os.path.exists('emotion_label_encoder.pkl'):
                self.label_encoder = joblib.load('emotion_label_encoder.pkl')
            else:
                # Create default encoder
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.emotions)
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def extract_features(self, audio_data, sr=22050):
        """Extract mel-spectrogram features"""
        try:
            # Ensure audio is 3 seconds
            target_length = sr * 3
            if len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
            else:
                audio_data = audio_data[:target_length]
            
            # Pre-emphasis filter
            pre_emphasis = 0.97
            audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, sr=sr, 
                n_mels=77, n_fft=2048, hop_length=512, win_length=2048
            )
            
            # Convert to dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = 2 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1
            
            # Pad/truncate to 174 time steps
            if mel_spec_norm.shape[1] < 174:
                pad_width = 174 - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_norm = mel_spec_norm[:, :174]
            
            return mel_spec_norm.T  # Shape: (174, 77)
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None
    
    def predict_emotion(self, audio_data, sr=22050):
        """Predict emotion from audio"""
        if self.model is None:
            return None, 0.0, {}
        
        try:
            # Extract features
            features = self.extract_features(audio_data, sr)
            if features is None:
                return None, 0.0, {}
            
            # Reshape for model: (174, 77) -> (1, 77, 174, 1)
            features_reshaped = features.T[np.newaxis, ..., np.newaxis]
            
            # Simple normalization
            features_norm = (features_reshaped - features_reshaped.mean()) / features_reshaped.std()
            
            # Predict
            predictions = self.model.predict(features_norm, verbose=0)
            
            # Get results
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_emotion = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            # All probabilities
            all_probs = {}
            for i, emotion in enumerate(self.label_encoder.classes_):
                all_probs[emotion] = float(predictions[0][i])
            
            return predicted_emotion, confidence, all_probs
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, 0.0, {}

def main():
    """Simple Streamlit application"""
    
    # Header
    st.title("🎭 Emotion Classification")
    st.write("Upload an audio file to analyze emotions")
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Load model
    if not classifier.load_model_components():
        st.error("Please ensure the model file 'final_emotion_model.h5' is present.")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file (WAV format recommended)"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        try:
            # Load audio
            audio_data, sr = librosa.load(temp_path, sr=22050)
            
            # Play audio
            st.audio(uploaded_file)
            
            # Predict button
            if st.button("🔮 Analyze Emotion", type="primary"):
                with st.spinner("Analyzing emotion..."):
                    predicted_emotion, confidence, all_probs = classifier.predict_emotion(audio_data, sr)
                
                if predicted_emotion:
                    # Main result
                    st.success(f"**Predicted Emotion: {predicted_emotion.upper()}**")
                    st.info(f"**Confidence: {confidence:.1%}**")
                    
                    # Show all probabilities
                    st.subheader("All Emotion Probabilities:")
                    for emotion, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{emotion.title()}**: {prob:.1%}")
                else:
                    st.error("Could not analyze the audio file.")
                
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    main()
