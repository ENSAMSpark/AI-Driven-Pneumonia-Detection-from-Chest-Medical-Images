"""
Fonctions utilitaires
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

def set_random_seed(seed=42):
    """
    Définit les graines aléatoires pour la reproductibilité
    """
    import random
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Graine aléatoire définie à: {seed}")

def plot_class_distribution(y_train, y_val, y_test):
    """
    Visualise la distribution des classes
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    datasets = [('Train', y_train), ('Validation', y_val), ('Test', y_test)]
    
    for i, (name, y) in enumerate(datasets):
        unique, counts = np.unique(y, return_counts=True)
        ax = axes[i]
        bars = ax.bar(['Pneumonia', 'Normal'], counts, 
                     color=['#ff6b6b', '#4ecdc4'])
        ax.set_title(f'Distribution - {name}')
        ax.set_ylabel('Nombre d\'images')
        
        # Ajouter les comptes sur les barres
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les pourcentages
    for name, y in datasets:
        total = len(y)
        pneumonia_count = np.sum(y == 0)
        normal_count = np.sum(y == 1)
        print(f"{name}:")
        print(f"  Pneumonia: {pneumonia_count} ({pneumonia_count/total*100:.1f}%)")
        print(f"  Normal:    {normal_count} ({normal_count/total*100:.1f}%)")

def visualize_augmented_images(images, labels, num_images=5):
    """
    Visualise des images augmentées
    """
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        ax = axes[i] if num_images > 1 else axes
        ax.imshow(images[i].squeeze(), cmap='gray')
        label = 'Pneumonia' if labels[i] == 0 else 'Normal'
        ax.set_title(f'Classe: {label}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results_to_file(results_dict, filename='results.txt'):
    """
    Sauvegarde les résultats dans un fichier
    """
    with open(filename, 'w') as f:
        for key, value in results_dict.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Résultats sauvegardés dans {filename}")

def print_model_summary_to_file(model, filename='model_summary.txt'):
    """
    Imprime le résumé du modèle dans un fichier
    """
    with open(filename, 'w') as f:
        # Rediriger la sortie de model.summary() vers le fichier
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Résumé du modèle sauvegardé dans {filename}")