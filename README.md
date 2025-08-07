# 🧠 AI Creator - Teachable Image Classifier

AI Creator est une application Python avec interface graphique (Tkinter) permettant d'entraîner un modèle de classification d’images **sans écrire une seule ligne de code**. Elle s'inspire du fonctionnement de Teachable Machine (Google), mais tourne localement sur votre ordinateur.

---

## 🚀 Fonctionnalités

- 📂 Chargement facile d’un dataset avec sous-dossiers par classe
- ➕ Ajout manuel de classes et d’images via l’interface
- ⚙️ Réglage des hyperparamètres (taille, batch, époques, patience)
- 🧠 Entraînement d’un modèle CNN avec TensorFlow
- 🧪 Test rapide sur une image
- 📤 Export en format `.tflite` pour une intégration mobile

---

## 🖼️ Structure attendue des données

Votre dossier d’images doit contenir un sous-dossier **par classe** :

/mon_dataset/
├── chats/
│ ├── chat1.jpg
│ ├── chat2.jpg
├── chiens/
│ ├── chien1.jpg
│ ├── chien2.jpg


## ⚙️ Installation

### 1. Cloner le dépôt


git clone https://github.com/votre-utilisateur/ai-creator.git
cd ai-creator

### 2. Créer un environnement virtuel (Python 3.10 recommandé)

python -m venv env
env\Scripts\activate  # Windows
### 3. Installer les dépendances

pip install -r requirements.txt
Si vous n'avez pas de requirements.txt, installez manuellement :

pip install tensorflow pillow numpy
### ▶️ Lancer l’application

python Ai_creator_V6.py
## 💻 Utilisation
Choisir un dossier d'images avec des classes

Ajuster les paramètres si nécessaire

Entraîner le modèle

Tester une image de votre choix

Exporter en .tflite pour le déploiement mobile

### ⚠️ Compatibilité GPU
TensorFlow utilise automatiquement le GPU si :

Vous avez une carte NVIDIA

CUDA 11.8 + cuDNN 8.6 sont installés

Les drivers sont à jour

### 📦 Export TensorFlow Lite
Une fois entraîné, le modèle peut être exporté en .tflite pour être utilisé sur mobile ou embarqué avec TensorFlow Lite.

## 📄 Licence
Ce projet est distribué sous licence apache 2.0.

## 🤝 Contribuer
Les contributions sont les bienvenues !
Forkez le projet, créez une branche, proposez vos améliorations par Pull Request ✨
