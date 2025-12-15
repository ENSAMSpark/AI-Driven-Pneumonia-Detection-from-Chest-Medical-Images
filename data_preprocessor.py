"""
Prétraitement et augmentation des données
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *
from sklearn.utils import class_weight

class DataPreprocessor:
    def __init__(self, augmentation_config=AUGMENTATION_CONFIG):
        self.augmentation_config = augmentation_config
        self.train_datagen = None
        self.val_datagen = None
        self.test_datagen = None
        
    def create_data_generators(self):
        """
        Crée des générateurs de données pour l'entraînement, validation et test
        """
        # Générateur pour l'entraînement (avec augmentation)
        self.train_datagen = ImageDataGenerator(
            **self.augmentation_config,
            rescale=1./255
        )
        
        # Générateur pour validation et test (sans augmentation)
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        return self.train_datagen, self.val_datagen, self.test_datagen
    
    def create_balanced_dataset(self, X, y, target_samples_per_class=None):
        """
        Crée un dataset équilibré en augmentant la classe minoritaire
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Distribution initiale: {dict(zip(unique_classes, class_counts))}")
        
        if target_samples_per_class is None:
            target_samples_per_class = max(class_counts)
        
        augmented_images = []
        augmented_labels = []
        
        for class_idx in unique_classes:
            class_indices = np.where(y == class_idx)[0]
            class_images = X[class_indices]
            current_count = len(class_images)
            
            if current_count < target_samples_per_class:
                # Augmenter la classe minoritaire
                needed = target_samples_per_class - current_count
                
                # Créer un générateur pour cette classe
                class_datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
                )
                
                # Augmenter les images
                class_images_expanded = np.expand_dims(class_images, axis=-1)
                augmented = []
                
                for batch in class_datagen.flow(class_images_expanded, 
                                               batch_size=len(class_images_expanded), 
                                               shuffle=False):
                    augmented.extend(batch)
                    if len(augmented) >= needed:
                        break
                
                augmented = np.squeeze(augmented[:needed], axis=-1)
                
                # Ajouter aux données
                augmented_images.extend(class_images)
                augmented_images.extend(augmented)
                augmented_labels.extend([class_idx] * (current_count + len(augmented)))
            else:
                # Sous-échantillonner la classe majoritaire si nécessaire
                sampled_indices = np.random.choice(
                    class_indices, 
                    size=min(target_samples_per_class, current_count), 
                    replace=False
                )
                augmented_images.extend(X[sampled_indices])
                augmented_labels.extend([class_idx] * len(sampled_indices))
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def calculate_class_weights(self, y):
        """
        Calcule les poids des classes pour l'entraînement
        """
        unique_classes = np.unique(y)
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y
        )
        return dict(zip(unique_classes, class_weights))