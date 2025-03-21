# live

# Liveness Detection Model for E-Identity & Verification

## 🚀 Project Overview
This project implements a **liveness detection model** to enhance security in **e-identity and verification systems** for blockchain-based applications. The model detects spoofing attacks such as **print, replay, 3D mask, and deepfake attacks** to ensure secure authentication.

<p align="center">
  <img src="https://github.com/Nexa-Bites/trustid_front_and_backend/blob/main/WhatsApp%20Image%202025-03-20%20at%2013.23.22_3df22ccf.png?raw=true" alt="TrustID Logo" width="250">
</p>

## 🛠️ Technologies Used
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, DeepFace
- **Machine Learning**: Scikit-learn
- **Backend**: 
- **Frontend**: 
- **Blockchain**: Smart Contracts (Ethereum)

## 📂 Project Structure
```
liveness-detection/
│-- model/            # Trained models & scripts
│-- api/              
│-- frontend/         
│-- datasets/         # Training datasets
│-- README.md         # Project documentation
│-- requirements.txt  # Dependencies
│-- .gitignore        # Ignore unnecessary files
```

## 📦 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone <repository_url>
cd liveness-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt  # Install Python dependencies
```

### 3️⃣ Run the Backend API


### 4️⃣ Run the Frontend 



## 🏗️ Model Training
To train the liveness detection model, use:
```bash
python train.py --dataset datasets/liveness --epochs 50
```

## 🔍 Model Evaluation
To test the model:
```bash
python evaluate.py --model model/liveness_detection_finetuned.h5
```

## 🔗 Blockchain Integration
- The verified liveness score is stored on the blockchain using **smart contracts**.
- This ensures **immutable** and **tamper-proof** identity verification.

## 📌 Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added new detection feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.





---
🔗 **Repository URL**: `https://github.com/Nexa-Bites/live/edit/main/README.md`

