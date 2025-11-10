# 🧠 Parkinson’s Voice Detection MVP

This project is an AI-powered web app that detects early signs of **Parkinson’s Disease** from **voice samples** using machine learning.  
It uses speech feature extraction (`librosa`) and an ensemble model (`XGBoost`, `RandomForest`, etc.) to predict whether a voice is **normal** or **Parkinson’s-affected**.

---

## 🚀 Features

- 🎤 Record or upload a voice sample (6 seconds)
- 🧩 Automatic feature extraction using `librosa`
- 🤖 Model predicts Parkinson’s probability
- 📊 Real-time frontend visualization
- 🧱 Built with a scalable backend + modern UI

---

## 🧬 Model Architecture

```
Input (voice/audio)  
    ↓  
Feature Extraction (librosa)
    ↓  
22 Features (Pitch, RMS, ZCR, MFCCs, etc.)
    ↓  
Scaling (StandardScaler)
    ↓  
XGBoost Classifier
    ↓  
Output → [0: Normal, 1: Parkinson’s Detected]
```

---

## 🗂️ Folder Structure

```
parkinsons_voice_mvp/
│
├── backend/
│   ├── main.py                 # FastAPI backend entry
│   ├── requirements.txt        # Python dependencies
│   ├── utils/
│   │   ├── __init__.py
│   │   └── feature_extraction.py
│   ├── models/
│   │   ├── xgb_model.pkl
│   │   └── scaler.pkl
│   └── ...
│
├── frontend/
│   ├── index.html              # Voice recording interface
│   ├── app.js                  # Handles audio + API calls
│   ├── styles.css              # UI styling
│   └── ...
│
├── models/
│   └── (Trained ML models)
│
├── README.md                   # Project documentation
└── .gitignore
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | FastAPI, Uvicorn |
| **ML/AI** | XGBoost, NumPy, Librosa, Scikit-learn |
| **Environment** | Python 3.11, Virtualenv |
| **Version Control** | Git + GitHub |

---

## 🧰 Installation & Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/parkinsons-voice-mvp.git
cd parkinsons-voice-mvp
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4️⃣ Run the backend
```bash
cd backend
uvicorn main:app --reload
```

Backend will start on → http://127.0.0.1:8000

### 5️⃣ Open the frontend
Just open `frontend/index.html` in your browser.

---

## 🧑‍💻 Contributing

Contributions are welcome! Follow these steps:

1. **Fork** the project  
2. **Create a new branch**
   ```bash
   git checkout -b feature-name
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add new feature"
   ```
4. **Push your branch**
   ```bash
   git push origin feature-name
   ```
5. **Open a Pull Request**

---

## 📜 License

This project is licensed under the **MIT License** — you’re free to use, modify, and share with attribution.

---

## 💡 Future Improvements

- Add CNN-based feature embeddings  
- Improve real-time voice noise filtering  
- Deploy to Hugging Face / Render  
- Add explainable AI visualizations

---

## 👨‍🔬 Author

**Vivek Nayi**  
📧 your.email@example.com  
🌐 https://github.com/YOUR_USERNAME

---

⭐ *If you found this helpful, consider starring the repo!*
