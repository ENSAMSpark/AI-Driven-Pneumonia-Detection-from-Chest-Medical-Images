"""
Évaluation du modèle
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)
from tensorflow.keras.models import load_model

class ModelEvaluator:
    def __init__(self, model=None, model_path=None):
        if model_path:
            self.model = load_model(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("Fournir soit un modèle soit un chemin vers un modèle")
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur le test set
        """
        print("Évaluation du modèle...")
        
        # Évaluation de base
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']
        for metric, value in zip(metrics, evaluation):
            print(f"{metric}: {value:.4f}")
        
        return evaluation
    
    def predict(self, X_test):
        """
        Prédit les probabilités et les classes
        """
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred_class = (y_pred_prob > 0.5).astype(int).flatten()
        
        return y_pred_prob, y_pred_class
    
    def compute_metrics(self, y_true, y_pred_class, y_pred_prob=None):
        """
        Calcule les métriques de classification
        """
        print("\n" + "="*50)
        print("RAPPORT DE CLASSIFICATION")
        print("="*50)
        
        # Classification report
        target_names = ['Pneumonia', 'Normal']
        print(classification_report(y_true, y_pred_class, target_names=target_names))
        
        # Métriques par classe
        precision = precision_score(y_true, y_pred_class, average=None)
        recall = recall_score(y_true, y_pred_class, average=None)
        f1 = f1_score(y_true, y_pred_class, average=None)
        
        for i, class_name in enumerate(target_names):
            print(f"\n{class_name}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall:    {recall[i]:.4f}")
            print(f"  F1-Score:  {f1[i]:.4f}")
        
        # Métriques globales
        print(f"\nMétriques globales:")
        print(f"  Accuracy:  {np.mean(y_true == y_pred_class):.4f}")
        print(f"  Precision (macro): {np.mean(precision):.4f}")
        print(f"  Recall (macro):    {np.mean(recall):.4f}")
        print(f"  F1-Score (macro):  {np.mean(f1):.4f}")
        
        if y_pred_prob is not None:
            # Courbe ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            print(f"\nAUC-ROC: {roc_auc:.4f}")
            
            # Courbe Precision-Recall
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
            pr_auc = auc(recall_curve, precision_curve)
            print(f"AUC-PR:  {pr_auc:.4f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc if y_pred_prob is not None else None,
            'pr_auc': pr_auc if y_pred_prob is not None else None
        }
    
    def plot_confusion_matrix(self, y_true, y_pred_class, save_path=None):
        """
        Affiche la matrice de confusion
        """
        cm = confusion_matrix(y_true, y_pred_class)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Pneumonia', 'Normal'],
                   yticklabels=['Pneumonia', 'Normal'])
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie étiquette')
        plt.xlabel('Prédiction')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Afficher les valeurs détaillées
        tn, fp, fn, tp = cm.ravel()
        print(f"\nDétail de la matrice de confusion:")
        print(f"True Negatives (Pneumonia correct):  {tn}")
        print(f"False Positives:                     {fp}")
        print(f"False Negatives:                     {fn}")
        print(f"True Positives (Normal correct):     {tp}")
    
    def plot_roc_curve(self, y_true, y_pred_prob, save_path=None):
        """
        Trace la courbe ROC
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Courbe ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_predictions(self, X_test, y_test, y_pred_class, y_pred_prob, 
                           num_samples=5, save_dir='./analysis'):
        """
        Analyse visuelle des prédictions
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Trouver des exemples de bonnes et mauvaises prédictions
        correct_indices = np.where(y_test == y_pred_class)[0]
        incorrect_indices = np.where(y_test != y_pred_class)[0]
        
        print(f"\nExemples d'analyse:")
        print(f"  Prédictions correctes: {len(correct_indices)}")
        print(f"  Prédictions incorrectes: {len(incorrect_indices)}")
        
        # Afficher quelques exemples
        fig, axes = plt.subplots(2, min(5, num_samples), figsize=(15, 6))
        
        # Correct predictions
        for i, idx in enumerate(correct_indices[:min(5, num_samples)]):
            ax = axes[0, i] if num_samples > 1 else axes[0]
            ax.imshow(X_test[idx].squeeze(), cmap='gray')
            true_label = 'Pneumonia' if y_test[idx] == 0 else 'Normal'
            pred_label = 'Pneumonia' if y_pred_class[idx] == 0 else 'Normal'
            prob = y_pred_prob[idx][0]
            ax.set_title(f'Vrai: {true_label}\nPrédit: {pred_label}\nProb: {prob:.3f}')
            ax.axis('off')
        
        # Incorrect predictions
        for i, idx in enumerate(incorrect_indices[:min(5, num_samples)]):
            ax = axes[1, i] if num_samples > 1 else axes[1]
            ax.imshow(X_test[idx].squeeze(), cmap='gray')
            true_label = 'Pneumonia' if y_test[idx] == 0 else 'Normal'
            pred_label = 'Pneumonia' if y_pred_class[idx] == 0 else 'Normal'
            prob = y_pred_prob[idx][0]
            ax.set_title(f'Vrai: {true_label}\nPrédit: {pred_label}\nProb: {prob:.3f}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()