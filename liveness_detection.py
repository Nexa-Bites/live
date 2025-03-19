# -*- coding: utf-8 -*-
"""Liveness_detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1C3mXo_LS6Fle8RjRniCRcmSUHR3w8iZg
"""


import kagglehub


path = kagglehub.dataset_download("trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1")

print("Path to dataset files:", path)

import os

dataset_path = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7"


files = os.listdir(dataset_path)
print("Dataset Files:", files)

import pandas as pd

csv_path = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7/ibeta_info.csv"

df = pd.read_csv(csv_path)
print(df.head())  
import os

dataset_folder = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7/rose_p3"

files = os.listdir(dataset_folder)

print("Sample files:", files[:10])

import os

folder_path = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7/rose_p3/real"


files = os.listdir(folder_path)


print("Sample files in 'real':", files[:10])

import os
 
user_folder_path = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7/rose_p3/real/user035"


files = os.listdir(user_folder_path)

print("Sample files in 'real/user035':", files[:10])

import os

device_folder_path = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7/rose_p3/real/user035/andr"

files = os.listdir(device_folder_path)

print("Sample files in 'real/user035/andr':", files[:10])

import cv2
import os

base_path = "/root/.cache/kagglehub/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1/versions/7/rose_p3"

real_video_path = os.path.join(base_path, "real")
spoof_types = ["mask", "mask3d", "monitor", "outline", "outline3d"]
spoof_video_paths = [os.path.join(base_path, spoof) for spoof in spoof_types]

real_output_folder = "extracted_real_frames"
spoof_output_folder = "extracted_spoof_frames"

os.makedirs(real_output_folder, exist_ok=True)
os.makedirs(spoof_output_folder, exist_ok=True)

def extract_frames(video_folder, output_folder):
    for user in os.listdir(video_folder):
        user_folder = os.path.join(video_folder, user)

        if os.path.isdir(user_folder):
            for device in os.listdir(user_folder):
                device_folder = os.path.join(user_folder, device)

                if os.path.isdir(device_folder):
                    for video_file in os.listdir(device_folder):
                        if video_file.endswith(".mp4"):
                            video_path = os.path.join(device_folder, video_file)
                            cap = cv2.VideoCapture(video_path)
                            frame_count = 0

                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                if frame_count % 10 == 0: 
                                    frame_filename = f"{user}_{device}_{video_file}_frame_{frame_count}.jpg"
                                    frame_filepath = os.path.join(output_folder, frame_filename)
                                    cv2.imwrite(frame_filepath, frame)

                                frame_count += 1

                            cap.release()


extract_frames(real_video_path, real_output_folder)
print(f"✅ Real face frames saved in: {real_output_folder}")


for spoof_path in spoof_video_paths:
    if os.path.exists(spoof_path):
        extract_frames(spoof_path, spoof_output_folder)

print(f"✅ Spoof attack frames saved in: {spoof_output_folder}")



import os
import cv2
import numpy as np





from tqdm import tqdm

real_input_folder = "extracted_real_frames"
spoof_input_folder = "extracted_spoof_frames"
preprocessed_real_folder = "preprocessed_real"
preprocessed_spoof_folder = "preprocessed_spoof"

os.makedirs(preprocessed_real_folder, exist_ok=True)
os.makedirs(preprocessed_spoof_folder, exist_ok=True)

IMG_SIZE = (224, 224)  

def preprocess_images(input_folder, output_folder):
    for image_file in tqdm(os.listdir(input_folder), desc=f"Processing {input_folder}"):
        img_path = os.path.join(input_folder, image_file)

        img = cv2.imread(img_path)
        if img is None:
            continue 


         
        img = cv2.resize(img, IMG_SIZE)

       
        img = img.astype(np.float32) / 255.0

        save_path = os.path.join(output_folder, image_file)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))  

        
preprocess_images(real_input_folder, preprocessed_real_folder)
preprocess_images(spoof_input_folder, preprocessed_spoof_folder)

print("✅ Image preprocessing complete! Preprocessed images saved.")

import os
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
import pickle

preprocessed_real_folder = "preprocessed_real"
preprocessed_spoof_folder = "preprocessed_spoof"

MODEL_NAME = "Facenet"

def extract_features(input_folder):
    features = []
    labels = []

    for image_file in tqdm(os.listdir(input_folder), desc=f"Extracting features from {input_folder}"):
        img_path = os.path.join(input_folder, image_file)

        try:
            
            embedding = DeepFace.represent(img_path, model_name=MODEL_NAME, enforce_detection=False)[0]['embedding']
            features.append(embedding)

           
            label = 0 if "real" in input_folder else 1
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(features), np.array(labels)

real_features, real_labels = extract_features(preprocessed_real_folder)
spoof_features, spoof_labels = extract_features(preprocessed_spoof_folder)

X = np.vstack((real_features, spoof_features))
y = np.hstack((real_labels, spoof_labels))

with open("features.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("✅ Feature extraction complete! Features saved in 'features.pkl'.")


import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


with open("features.pkl", "rb") as f:
    X, y = pickle.load(f)


X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"✅ SVM Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)


accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"✅ Random Forest Accuracy: {accuracy_rf:.4f}")
print(classification_report(y_test, y_pred_rf))

import pickle
import numpy as np


with open("features.pkl", "rb") as f:
    X, y = pickle.load(f)

print(f"✅ Loaded features: {X.shape}, Labels: {y.shape}")


X = X / np.max(X)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)), 
    layers.Dropout(0.3),  
    layers.Dense(64, activation='relu'),  
    layers.Dense(1, activation='sigmoid') 
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=16, validation_data=(X_test_scaled, y_test))


test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

import matplotlib.pyplot as plt


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()

model.save("liveness_model.h5")

from tensorflow.keras.models import load_model

model = load_model("liveness_model.h5")

model.summary()  

y_pred = model.predict(X_test_scaled)

new_image_path = "SIHU.JPG"
new_embedding = DeepFace.represent(new_image_path, model_name='Facenet', enforce_detection=False)[0]['embedding']
new_embedding_scaled = scaler.transform([new_embedding])
prediction = model.predict(new_embedding_scaled)
print("Real" if prediction < 0.5 else "Spoof")

print(prediction)



from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,  
    width_shift_range=0.2, 
    height_shift_range=0.2,  
    horizontal_flip=True, 
)

train_generator = datagen.flow(X_train_scaled, y_train, batch_size=32)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)


class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}


model.fit(X_train_scaled, y_train, class_weight=class_weight_dict, batch_size=32, epochs=10)

from tensorflow.keras import regularizers

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1
)

from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=20,
    class_weight=class_weight_dict,  
    callbacks=[lr_scheduler] 
)

model.save("liveness_model_finetuned.h5")

from google.colab import files
files.download("liveness_model_finetuned.h5")

