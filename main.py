"""
Script principal pour exécuter tout le pipeline
"""

import numpy as np
import os
from config import *
from data_loader import load_data, prepare_data
from data_preprocessor import DataPreprocessor
from model_builder import PneumoniaCNN
from train import ModelTrainer
from evaluate import ModelEvaluator
from utils import set_random_seed, plot_class_distribution, save_results_to_file

def main():
    """
    Pipeline complet de détection de pneumonie
    """
    # 1. Initialisation
    print("="*60)
    print("DÉTECTION DE PNEUMONIE PAR CNN")
    print("="*60)
    
    set_random_seed(42)
    
    # 2. Chargement des données
    print("\n1. CHARGEMENT DES DONNÉES")
    X_train, y_train, X_test, y_test = load_data()
    
    # 3. Prétraitement des données
    print("\n2. PRÉTRAITEMENT DES DONNÉES")
    preprocessor = DataPreprocessor()
    
    # Créer un dataset équilibré
    X_train_balanced, y_train_balanced = preprocessor.create_balanced_dataset(
        X_train, y_train
    )
    
    # Diviser en train/validation
    X_train_final, X_val, y_train_final, y_val = prepare_data(
        X_train_balanced, y_train_balanced
    )
    
    # Visualiser la distribution
    plot_class_distribution(y_train_final, y_val, y_test)
    
    # Calculer les poids des classes
    class_weights = preprocessor.calculate_class_weights(y_train_final)
    print(f"\nPoids des classes: {class_weights}")
    
    # Créer les générateurs de données
    train_datagen, val_datagen, _ = preprocessor.create_data_generators()
    
    # Configurer les générateurs
    train_generator = train_datagen.flow(
        X_train_final, y_train_final,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # 4. Construction du modèle
    print("\n3. CONSTRUCTION DU MODÈLE")
    model_builder = PneumoniaCNN()
    model = model_builder.build_model()
    model = model_builder.compile_model()
    model_builder.summary()
    
    # 5. Entraînement du modèle
    print("\n4. ENTRAÎNEMENT DU MODÈLE")
    trainer = ModelTrainer(model)
    history = trainer.train(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weights=class_weights
    )
    
    # Visualiser l'entraînement
    trainer.plot_training_history(save_path='./saved_models/training_history.png')
    trainer.save_history(HISTORY_PATH)
    
    # 6. Sauvegarde du modèle
    print("\n5. SAUVEGARDE DU MODÈLE")
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé: {MODEL_PATH}")
    
    # 7. Évaluation du modèle
    print("\n6. ÉVALUATION DU MODÈLE")
    evaluator = ModelEvaluator(model_path='best_model.keras')
    
    # Évaluation de base
    evaluation = evaluator.evaluate(X_test, y_test)
    
    # Prédictions
    y_pred_prob, y_pred_class = evaluator.predict(X_test)
    
    # Calcul des métriques
    metrics = evaluator.compute_metrics(y_test, y_pred_class, y_pred_prob)
    
    # Visualisations
    evaluator.plot_confusion_matrix(y_test, y_pred_class, 
                                   save_path='./saved_models/confusion_matrix.png')
    evaluator.plot_roc_curve(y_test, y_pred_prob,
                            save_path='./saved_models/roc_curve.png')
    
    # Analyse détaillée
    evaluator.analyze_predictions(X_test, y_test, y_pred_class, y_pred_prob,
                                 num_samples=5, save_dir='./analysis')
    
    # 8. Sauvegarde des résultats
    print("\n7. SAUVEGARDE DES RÉSULTATS")
    results = {
        'test_metrics': {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'precision': evaluation[2],
            'recall': evaluation[3],
            'auc': evaluation[4]
        },
        'detailed_metrics': metrics,
        'training_history': {
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
    }
    
    save_results_to_file(results, './saved_models/final_results.txt')
    
    print("\n" + "="*60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    
    return model, history, results

if __name__ == "__main__":
    # Créer les répertoires nécessaires
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./analysis', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Exécuter le pipeline
    model, history, results = main()