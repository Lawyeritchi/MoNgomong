"""
Train BISINDO model dari CSV yang sudah ada
Data sudah dalam format landmarks, tinggal train!
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data dari CSV files"""
    print("📂 Loading data dari CSV files...")
    
    # Load datasets
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"   ✅ Train set: {len(train_df)} samples")
    print(f"   ✅ Val set: {len(val_df)} samples")
    print(f"   ✅ Test set: {len(test_df)} samples")
    print(f"   ✅ Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    """Prepare X and y dari dataframe"""
    print("\n📊 Preparing data...")
    
    # Assume last column is label, rest are features
    # Check column names
    print(f"   Columns: {train_df.columns.tolist()[:5]}... (showing first 5)")
    
    # Identify label column (biasanya 'label', 'class', atau nama huruf)
    possible_label_cols = ['label', 'class', 'target', 'letter']
    label_col = None
    
    for col in possible_label_cols:
        if col in train_df.columns:
            label_col = col
            break
    
    # Kalau gak ada, assume kolom terakhir
    if label_col is None:
        label_col = train_df.columns[-1]
    
    print(f"   Label column: '{label_col}'")
    
    # Separate features and labels
    X_train = train_df.drop(columns=[label_col]).values
    y_train = train_df[label_col].values
    
    X_val = val_df.drop(columns=[label_col]).values
    y_val = val_df[label_col].values
    
    X_test = test_df.drop(columns=[label_col]).values
    y_test = test_df[label_col].values
    
    print(f"   ✅ Features shape: {X_train.shape[1]} features")
    print(f"   ✅ Classes: {sorted(np.unique(y_train))}")
    print(f"   ✅ Number of classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("\n🤖 Training Random Forest model...")
    print("   (Ini mungkin butuh 1-3 menit tergantung data size)")
    
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
    
    # Detailed classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return accuracy

def save_model(model, filename='model/rf_bisindo_99.pkl'):
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

def test_model_loading(filename='model/rf_bisindo_99.pkl'):
    """Test loading model kembali"""
    print(f"\n🔍 Testing model loading...")
    
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
        
        print("   ✅ Model loaded successfully!")
        print(f"   📊 Model type: {type(loaded_model).__name__}")
        print(f"   🔢 Features: {loaded_model.n_features_in_}")
        print(f"   🏷️  Classes: {len(loaded_model.classes_)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def main():
    print("="*60)
    print("🚀 BISINDO Model Training dari CSV")
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
        print(f"✅ Model saved: model/rf_bisindo_99.pkl")
        print(f"✅ Features: {model.n_features_in_}")
        print(f"✅ Classes: {len(model.classes_)} ({', '.join(map(str, sorted(model.classes_)))})")
        
        if load_success:
            print("\n📝 Next step: Jalankan Flask API")
            print("   python3 app.py")
        else:
            print("\n⚠️  Model tersimpan tapi ada issue saat loading")
            print("   Coba jalankan app.py dan lihat hasilnya")
        
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ File tidak ditemukan: {e}")
        print("\n💡 Pastikan file CSV ada di folder data/:")
        print("   - data/train.csv")
        print("   - data/val.csv")
        print("   - data/test.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training dibatalkan oleh user")