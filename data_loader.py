"""
Chargement et préparation des données
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

def load_images_from_folder(folder_path, img_size=IMG_SIZE):
    """
    Charge les images depuis un dossier avec sous-dossiers de classes
    """
    images = []
    labels = []
    class_names = ['PNEUMONIA', 'NORMAL']
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        
        if not os.path.exists(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            try:
                # Lire l'image en niveaux de gris
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                    
                # Redimensionner
                img = cv2.resize(img, (img_size, img_size))
                
                # Normaliser
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {e}")
                continue
    
    return np.array(images), np.array(labels)

def load_data():
    """
    Charge toutes les données d'entraînement et de test
    """
    print("Chargement des données d'entraînement...")
    X_train, y_train = load_images_from_folder(TRAIN_DIR)
    
    print("Chargement des données de test...")
    X_test, y_test = load_images_from_folder(TEST_DIR)
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Reshape pour CNN
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    
    return X_train, y_train, X_test, y_test

def prepare_data(X_train, y_train, test_size=0.2, random_state=42):
    """
    Divise les données en train/validation
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_train  # Maintient la distribution des classes
    )
    
    print(f"Train split: {X_train.shape}")
    print(f"Validation split: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val