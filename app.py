"""
Flask API Backend untuk SignSpeak Web
Real-time BISINDO Classification
Support 2 models: Huruf (letters) dan Kata (words)

Endpoints:
- GET  /api/health           : Check API status
- GET  /api/model-info       : Get models information
- POST /api/predict          : Predict letter (1 hand, 126 features)
- POST /api/predict-word     : Predict word (1-2 hands, 126 features)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# CORS configuration for production
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "https://*.netlify.app",
            "https://*.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables untuk models
model_letter = None
model_word = None
models_info = {}

def load_models():
    """Load both models (letter + word)"""
    global model_letter, model_word, models_info
    
    print("📂 Loading models...")
    
    # Load Letter Model
    letter_path = 'model/rf_bisindo_99.pkl'
    if os.path.exists(letter_path):
        with open(letter_path, 'rb') as f:
            model_letter = pickle.load(f)
        
        models_info['letter'] = {
            'model_type': type(model_letter).__name__,
            'n_features': model_letter.n_features_in_,
            'n_classes': len(model_letter.classes_),
            'classes': model_letter.classes_.tolist(),
            'model_path': letter_path
        }
        print(f"✅ Letter Model loaded")
        print(f"   - Features: {models_info['letter']['n_features']}")
        print(f"   - Classes: {models_info['letter']['n_classes']} letters")
    else:
        print(f"⚠️  Letter model not found: {letter_path}")
    
    # Load Word Model
    word_path = 'model/rf_bisindo_words.pkl'
    if os.path.exists(word_path):
        with open(word_path, 'rb') as f:
            model_word = pickle.load(f)
        
        models_info['word'] = {
            'model_type': type(model_word).__name__,
            'n_features': model_word.n_features_in_,
            'n_classes': len(model_word.classes_),
            'classes': model_word.classes_.tolist(),
            'model_path': word_path
        }
        print(f"✅ Word Model loaded")
        print(f"   - Features: {models_info['word']['n_features']}")
        print(f"   - Classes: {models_info['word']['n_classes']} words")
    else:
        print(f"⚠️  Word model not found: {word_path}")
    
    if model_letter is None and model_word is None:
        raise FileNotFoundError("No models found!")

# Load models saat startup
try:
    load_models()
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️  API akan jalan tapi predict tidak akan work")

# ============= ROUTES =============

@app.route('/')
def home():
    """Homepage - API info"""
    return jsonify({
        'status': 'online',
        'app': 'SignSpeak API',
        'version': '2.0.0',
        'description': 'Real-time BISINDO Classification API - Letters & Words',
        'endpoints': {
            'health': '/api/health',
            'model_info': '/api/model-info',
            'predict_letter': '/api/predict (POST)',
            'predict_word': '/api/predict-word (POST)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'letter_model_loaded': model_letter is not None,
        'word_model_loaded': model_word is not None
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get models information"""
    return jsonify({
        'status': 'success',
        'models': models_info
    })

@app.route('/api/predict', methods=['POST'])
def predict_letter():
    """
    Predict BISINDO letter dari hand landmarks (1 hand)
    
    Expected JSON input:
    {
        "landmarks": [x1, y1, z1, ..., x21, y21, z21] (126 values)
    }
    """
    
    if model_letter is None:
        return jsonify({
            'status': 'error',
            'message': 'Letter model not loaded'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'landmarks' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing landmarks data'
            }), 400
        
        landmarks = data['landmarks']
        expected_features = models_info['letter']['n_features']
        
        if len(landmarks) != expected_features:
            return jsonify({
                'status': 'error',
                'message': f'Invalid features. Expected {expected_features}, got {len(landmarks)}'
            }), 400
        
        # Convert & predict
        landmarks_array = np.array(landmarks).reshape(1, -1)
        prediction = model_letter.predict(landmarks_array)[0]
        probabilities = model_letter.predict_proba(landmarks_array)[0]
        
        # Get confidence
        predicted_idx = np.where(model_letter.classes_ == prediction)[0][0]
        confidence = float(probabilities[predicted_idx])
        
        # All probabilities
        all_probs = {
            str(label): float(prob) 
            for label, prob in zip(model_letter.classes_, probabilities)
        }
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'status': 'success',
            'mode': 'letter',
            'prediction': {
                'letter': str(prediction),
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 2),
                'all_probabilities': sorted_probs,
                'top_3': dict(list(sorted_probs.items())[:3])
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/predict-word', methods=['POST'])
def predict_word():
    """
    Predict BISINDO word dari hand landmarks (1-2 hands)
    
    Expected JSON input:
    {
        "landmarks": [hand1_x1, hand1_y1, hand1_z1, ..., hand2_x21, hand2_y21, hand2_z21]
    }
    
    Note: Bisa 126 features (1 hand) atau 252 features (2 hands)
    """
    
    if model_word is None:
        return jsonify({
            'status': 'error',
            'message': 'Word model not loaded'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'landmarks' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing landmarks data'
            }), 400
        
        landmarks = data['landmarks']
        expected_features = models_info['word']['n_features']
        
        # Handle both 126 (1 hand) and 252 (2 hands) features
        if len(landmarks) == 126 and expected_features == 126:
            # Model expects 126, got 126 - OK
            pass
        elif len(landmarks) == 252 and expected_features == 126:
            # Model expects 126, got 252 - take first hand only
            landmarks = landmarks[:126]
        elif len(landmarks) == 126 and expected_features == 252:
            # Model expects 252, got 126 - duplicate
            landmarks = landmarks + landmarks
        elif len(landmarks) != expected_features:
            return jsonify({
                'status': 'error',
                'message': f'Invalid features. Expected {expected_features}, got {len(landmarks)}'
            }), 400
        
        # Convert & predict
        landmarks_array = np.array(landmarks).reshape(1, -1)
        prediction = model_word.predict(landmarks_array)[0]
        probabilities = model_word.predict_proba(landmarks_array)[0]
        
        # Get confidence
        predicted_idx = np.where(model_word.classes_ == prediction)[0][0]
        confidence = float(probabilities[predicted_idx])
        
        # Clean label (remove 'kata_' prefix)
        word_clean = prediction.replace('kata_', '').replace('_', ' ').upper()
        
        # All probabilities (cleaned)
        all_probs = {
            label.replace('kata_', '').replace('_', ' ').upper(): float(prob) 
            for label, prob in zip(model_word.classes_, probabilities)
        }
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'status': 'success',
            'mode': 'word',
            'prediction': {
                'word': word_clean,
                'word_raw': str(prediction),
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 2),
                'all_probabilities': sorted_probs,
                'top_3': dict(list(sorted_probs.items())[:3])
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============= RUN APP =============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SignSpeak API Server v2.0")
    print("="*60)
    print()
    
    # Status
    if model_letter is not None:
        print(f"✅ Letter Model: {models_info['letter']['n_classes']} classes")
    else:
        print("⚠️  Letter Model not loaded")
    
    if model_word is not None:
        print(f"✅ Word Model: {models_info['word']['n_classes']} classes")
    else:
        print("⚠️  Word Model not loaded")
    
    print()
    print("🌐 Starting server...")
    print("📍 API running at: http://localhost:5000")
    print()
    print("📝 Available endpoints:")
    print("   GET  /                  - API info")
    print("   GET  /api/health        - Health check")
    print("   GET  /api/model-info    - Models information")
    print("   POST /api/predict       - Letter prediction")
    print("   POST /api/predict-word  - Word prediction")
    print()
    print("💡 Press CTRL+C to stop")
    print("="*60)
    print()
    
    # Run
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )