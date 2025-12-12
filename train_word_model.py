"""
Script untuk train model kata BISINDO dari CSV
Similar dengan train_from_csv.py tapi untuk kata (2 hands, 252 features)

Usage:
    python3 train_word_model.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data dari CSV files"""
    print("📂 Loading word dataset dari CSV files...")
    
    train_df = pd.read_csv('data/words_processed/train_words.csv')
    val_df = pd.read_csv('data/words_processed/val_words.csv')
    test_df = pd.read_csv('data/words_processed/test_words.csv')
    
    print(f"   ✅ Train set: {len(train_df)} samples")
    print(f"   ✅ Val set: {len(val_df)} samples")
    print(f"   ✅ Test set: {len(test_df)} samples")
    print(f"   ✅ Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    """Prepare X and y dari dataframe"""
    print("\n📊 Preparing data...")
    
    label_col = 'label'
    
    print(f"   Label column: '{label_col}'")
    
    # Separate features and labels
    X_train = train_df.drop(columns=[label_col]).values
    y_train = train_df[label_col].values
    
    X_val = val_df.drop(columns=[label_col]).values
    y_val = val_df[label_col].values
    
    X_test = test_df.drop(columns=[label_col]).values
    y_test = test_df[label_col].values
    
    print(f"   ✅ Features shape: {X_train.shape[1]} features (252 = 2 hands)")
    print(f"   ✅ Classes: {len(np.unique(y_train))} words")
    print(f"   ✅ Word list: {sorted(np.unique(y_train))[:10]}... (showing first 10)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("\n🤖 Training Random Forest model for words...")
    print("   (Ini mungkin butuh 2-5 menit tergantung data size)")
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Train
    print("   🔄 Training in progress...")
    model.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    # Validate
    print("\n📊 Validation results:")
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"   ✅ Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n🧪 Testing model on test set...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Test Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed classification report (top 20 classes)
    print("\n📋 Classification Report (sample):")
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Print first 30 lines (top classes)
    for i, line in enumerate(report.split('\n')[:30]):
        print(line)
    
    return accuracy

def save_model(model, filename='model/rf_bisindo_words.pkl'):
    """Save model to file"""
    print(f"\n💾 Saving model to: {filename}")
    
    # Create directory if not exists
    Path(filename).parent.mkdir(exist_ok=True)
    
    # Save model
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("   ✅ Model saved successfully!")
    
    # File info
    import os
    file_size = os.path.getsize(filename) / 1024
    print(f"   📏 File size: {file_size:.2f} KB")

def test_model_loading(filename='model/rf_bisindo_words.pkl'):
    """Test loading model kembali"""
    print(f"\n🔍 Testing model loading...")
    
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
        
        print("   ✅ Model loaded successfully!")
        print(f"   📊 Model type: {type(loaded_model).__name__}")
        print(f"   🔢 Features: {loaded_model.n_features_in_}")
        print(f"   🏷️  Classes: {len(loaded_model.classes_)}")
        print(f"   📝 Sample classes: {loaded_model.classes_[:10]}... (first 10)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def main():
    print("="*60)
    print("🚀 BISINDO Word Model Training")
    print("="*60)
    print()
    
    try:
        # 1. Load data
        train_df, val_df, test_df = load_data()
        
        # 2. Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
            train_df, val_df, test_df
        )
        
        # 3. Train model
        model = train_model(X_train, y_train, X_val, y_val)
        
        # 4. Evaluate on test set
        test_accuracy = evaluate_model(model, X_test, y_test)
        
        # 5. Save model
        save_model(model)
        
        # 6. Test loading
        load_success = test_model_loading()
        
        # Summary
        print("\n" + "="*60)
        print("🎉 Training Complete!")
        print("="*60)
        print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"✅ Model saved: model/rf_bisindo_words.pkl")
        print(f"✅ Features: {model.n_features_in_} (2 hands)")
        print(f"✅ Classes: {len(model.classes_)} words")
        
        if load_success:
            print("\n📝 Next step: Update Flask API untuk support word detection")
        else:
            print("\n⚠️  Model tersimpan tapi ada issue saat loading")
        
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ File tidak ditemukan: {e}")
        print("\n💡 Pastikan sudah run prepare_word_dataset.py dulu!")
        print("   python3 prepare_word_dataset.py --input data/words_raw")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training dibatalkan oleh user")