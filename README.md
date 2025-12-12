# MoNgomong

Website penerjemah bahasa isyarat Indonesia (BISINDO) secara real-time menggunakan kamera. Sistem mendeteksi gesture tangan dan menerjemahkannya menjadi huruf alfabet.

## ✨ Fitur

- 🎥 **Real-time Detection**: Deteksi gesture tangan secara langsung dari kamera
- 🔤 **26 Huruf BISINDO**: Mendukung alfabet A-Z
- 📊 **Confidence Score**: Menampilkan tingkat kepercayaan prediksi
- 📝 **Translation Output**: Hasil terjemahan langsung tersimpan
- 📜 **History**: Riwayat 5 prediksi terakhir
- 🌓 **Dark/Light Mode**: Tema gelap dan terang
- 📱 **Mobile Responsive**: Bisa diakses dari HP
- ⚡ **Offline-Capable**: Berjalan di localhost tanpa internet

## 🛠️ Teknologi

**Frontend:**
- HTML5, CSS3, JavaScript (Vanilla)
- MediaPipe Hands (Hand Detection)
- Canvas API (Landmark Visualization)

**Backend:**
- Python 3.8+
- Flask (Web API)
- scikit-learn (Machine Learning)
- Random Forest Classifier (99% Accuracy)

## 📁 Struktur Project

```
realtime-bisindo-classification-main/
├── frontend/                   # Website frontend
│   ├── index.html             # Halaman utama
│   ├── css/
│   │   └── style.css          # Styling
│   └── js/
│       ├── config.js          # Konfigurasi
│       ├── api.js             # API handler
│       ├── mediapipe.js       # Hand detection
│       ├── ui.js              # UI manager
│       └── app.js             # Main application
├── model/                      # Machine learning model
│   └── rf_bisindo_99.pkl      # Trained Random Forest model
├── data/                       # Dataset (CSV format)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── app.py                      # Flask API backend
├── train_from_csv.py          # Script untuk retrain model
├── requirements.txt           # Python dependencies
└── README.md                  # Dokumentasi ini
```

## 🚀 Quick Start

### Prerequisites

Pastikan sudah terinstall:
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **pip** (biasanya sudah include dengan Python)
- **Web Browser** (Chrome/Firefox recommended)
- **Webcam** (built-in atau eksternal)

### Installation

**1. Clone/Download Repository**

```bash
# Clone via git
git clone https://github.com/KrisnaSantosa15/realtime-bisindo-classification.git
cd realtime-bisindo-classification-main

# Atau download ZIP dan extract
```

**2. Buat Virtual Environment**

```bash
# Buat venv
python3 -m venv venv

# Aktifkan venv
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install semua dependencies
pip install -r requirements.txt
```

**4. Jalankan Backend API**

```bash
# Pastikan masih di folder root dan venv aktif
python3 app.py
```

Output yang diharapkan:
```
============================================================
🚀 SignSpeak API Server
============================================================
✅ Model loaded successfully
📊 Ready to classify 26 BISINDO letters
🌐 Starting server...
📍 API akan running di: http://localhost:5000
```

**5. Jalankan Frontend (Terminal Baru)**

```bash
# Buka terminal baru
cd realtime-bisindo-classification-main/frontend

# Start HTTP server
python3 -m http.server 8000
```

**6. Buka di Browser**

Buka browser dan akses: **http://localhost:8000**

## 📖 Cara Penggunaan

### Langkah 1: Start Camera
1. Klik tombol **"Start Camera"**
2. Izinkan browser mengakses kamera
3. Tunggu video muncul

### Langkah 2: Start Detection
1. Klik tombol **"Start Detection"**
2. Status akan berubah menjadi "Active ✓"

### Langkah 3: Buat Gesture
1. Posisikan **1 TANGAN** di depan kamera
2. Buat gesture huruf BISINDO (A-Z)
3. Tahan gesture 2-3 detik
4. Sistem akan mendeteksi dan menampilkan hasil

### Langkah 4: Lihat Hasil
- **Hasil Deteksi**: Huruf yang terdeteksi + confidence score
- **Terjemahan**: Huruf-huruf terakumulasi di bagian "Terjemahan"
- **Riwayat**: 5 prediksi terakhir dengan confidence score

### Langkah 5: Stop
- Klik **"Pause Detection"** untuk pause sementara
- Klik **"Stop"** untuk stop kamera

## 💡 Tips untuk Accuracy Tinggi

1. **Pencahayaan**: Pastikan ruangan cukup terang
2. **Background**: Gunakan background polos (putih/hijau lebih bagus)
3. **Jarak**: Jangan terlalu dekat/jauh dari kamera
4. **Stabilitas**: Tahan gesture 2-3 detik, jangan terlalu cepat
5. **Fokus**: Hanya gunakan 1 tangan
6. **Posisi**: Tangan di tengah frame kamera

## 🔧 Troubleshooting

### Backend API tidak jalan

**Problem:** Error saat `python3 app.py`

**Solution:**
```bash
# Cek venv aktif atau belum
which python  # Mac/Linux
where python  # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Cek model file ada
ls -la model/rf_bisindo_99.pkl
```

### Kamera tidak muncul

**Problem:** Browser tidak bisa akses kamera

**Solution:**
1. Pastikan kamera tidak digunakan aplikasi lain (Zoom, Teams, dll)
2. Check permission browser (Settings > Privacy > Camera)
3. Coba browser lain (Chrome recommended)
4. Restart browser

### Prediction tidak muncul

**Problem:** Detection jalan tapi hasil tidak tampil

**Solution:**
1. Buka **Console Browser (F12)**
2. Lihat error message
3. Pastikan API status "Connected ✓"
4. Cek network tab - harusnya ada POST request ke `/api/predict`

### Confidence rendah (< 50%)

**Problem:** Accuracy prediksi jelek

**Solution:**
1. Perbaiki pencahayaan
2. Gunakan background polos
3. Tahan gesture lebih lama (3 detik)
4. Posisikan tangan lebih jelas
5. Coba gesture yang lebih jelas (A, B, C dulu)

### CORS Error

**Problem:** Browser block API request

**Solution:**
- Sudah di-handle dengan `flask-cors`
- Kalau masih error, restart backend API

## 🎓 Dataset & Model

### Model Information
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 99.49% (test set)
- **Features**: 126 (21 landmarks × 3 coords × 2 hands)
- **Classes**: 26 (A-Z BISINDO letters)
- **Training Data**: 10,145 samples

### Retrain Model (Optional)

Kalau mau retrain model dengan data baru:

```bash
# Pastikan venv aktif dan di folder root
python3 train_from_csv.py
```

Dataset format:
- CSV files di folder `data/`
- Columns: `label, hand_0_x, hand_0_y, hand_0_z, ..., hand_20_z`

## 📱 Mobile Support

Website responsive dan bisa diakses dari HP:

1. Pastikan HP dan laptop di **jaringan WiFi yang sama**
2. Cari IP address laptop:
   ```bash
   # Mac/Linux
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```
3. Buka browser di HP: `http://[IP_ADDRESS]:8000`
   - Contoh: `http://192.168.1.100:8000`

## 📝 Todo List

- [ ] Tambah mode **Kata/Kalimat**
- [ ] Tambah **Text-to-Speech** Bahasa Indonesia
- [ ] Support **2 tangan**
- [ ] Tambah **dataset kata-kata**
- [ ] Export terjemahan ke **file**
- [ ] Tambah **tutorial gesture** interaktif
- [ ] Optimisasi model untuk **mobile browser**

## 👥 Credits

- **Original Dataset**: [Alfabet BISINDO Kaggle](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)
- **MediaPipe**: Google MediaPipe Hands
- **Flask**: Pallets Projects
- **scikit-learn**: scikit-learn developers

## 📧 Contact

Untuk pertanyaan atau feedback:

---

**Built with ❤️ **