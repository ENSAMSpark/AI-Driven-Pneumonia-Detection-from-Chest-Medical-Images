
import subprocess
import sys

def install_requirements():
    """Installe les dépendances si nécessaire"""
    print("Installation des dépendances...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    """Exécute le pipeline complet"""
    print("Démarrage du pipeline de détection de pneumonie...")
    
    # Vérifier si les dépendances sont installées
    try:
        import tensorflow
        import opencv_python
        import sklearn
    except ImportError:
        print("Dépendances manquantes. Installation...")
        install_requirements()
    
    # Importer et exécuter le pipeline principal
    from main import main as run_pipeline
    
    # Exécuter le pipeline
    run_pipeline()

if __name__ == "__main__":
    main()