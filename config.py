"""
Configuration du projet - Hyperparamètres et chemins
"""

import os

# Chemins des données
DATA_DIR = './chest_xray'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Paramètres d'image
IMG_SIZE = 150
IMG_CHANNELS = 1  # Grayscale
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# Paramètres d'entraînement
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Paramètres d'augmentation de données
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

# Paramètres du modèle
MODEL_CONFIG = {
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'dropout': 0.1},
        {'filters': 64, 'kernel_size': (3, 3), 'dropout': 0.1},
        {'filters': 128, 'kernel_size': (3, 3), 'dropout': 0.2},
        {'filters': 128, 'kernel_size': (3, 3), 'dropout': 0.2},
    ],
    'dense_layers': [128],
    'output_activation': 'sigmoid'
}

# Callbacks
CALLBACKS_CONFIG = {
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.2,
        'patience': 3,
        'min_lr': 0.00001,
        'verbose': 1
    },
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True,
        'verbose': 1
    },
    'model_checkpoint': {
        'filepath': 'best_model.keras',
        'monitor': 'val_loss',
        'save_best_only': True,
        'verbose': 1
    }
}

# Chemins de sauvegarde
SAVE_DIR = './saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, 'pneumonia_cnn_model.h5')
HISTORY_PATH = os.path.join(SAVE_DIR, 'training_history.npy')