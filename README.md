# 🧠 AI Creator - Teachable Image Classifier (v6)

**AI Creator** est une application de bureau en Python dotée d'une interface graphique (Tkinter) qui permet à n'importe qui d'entraîner un modèle d’intelligence artificielle pour la classification d’images **de manière visuelle et intuitive**, sans écrire une seule ligne de code.

Inspirée de [Teachable Machine](https://teachablemachine.withgoogle.com/) de Google, cette version locale garantit confidentialité, personnalisation et flexibilité, tout en restant simple à utiliser.

---
## 🎯 Objectif du projet

**AI Creator** vise à démocratiser l’intelligence artificielle en permettant à tout utilisateur, même sans compétence en programmation, d’entraîner un modèle de classification d’images à partir de ses propres données, en quelques clics.

Ce projet permet :

- aux **débutants** de découvrir le machine learning de manière visuelle,
- aux **enseignants** de proposer des activités interactives en classe,
- aux **développeurs** de prototyper rapidement un modèle de vision par ordinateur **localement**.

---

## 🖥️ Interface utilisateur

L'application dispose d’une interface intuitive conçue avec **Tkinter**, divisée en deux sections :

### 🔹 Partie gauche – Gestion des classes

- Affichage des classes détectées
- Bouton « Ajouter une classe » (import manuel d’images)

### 🔹 Partie droite – Contrôle du modèle

- Sélection du dossier d’images
- Réglage des hyperparamètres : taille, batch, époques, patience
- Barres de progression + journal d’entraînement en direct
- Bouton **« Tester une image »**
- Bouton **« Exporter en TensorFlow Lite »**

---

## 🧠 Détails techniques

Le modèle est un **CNN simple** implémenté avec **Keras** et **TensorFlow** :

**Architecture :**
```
Input → Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(128) → Dense(n_classes, softmax)
```

- **Fonctions d’activation :** ReLU (couches cachées), Softmax (sortie)
- **Perte :** `categorical_crossentropy`
- **Optimiseur :** `Adam`
- **Métrique :** `accuracy`
- **EarlyStopping** sur la perte de validation

---

## ⚙️ Personnalisation

L'utilisateur peut configurer :

- La taille des images (largeur x hauteur)
- La taille du batch
- Le nombre d’époques
- Le seuil de patience (EarlyStopping)

---

## 🧪 Fonction de test

Une fois le modèle entraîné, l’utilisateur peut :

- Charger une image
- Obtenir instantanément la prédiction
- Voir le nom de la classe prédite

---

## 📤 Export en TensorFlow Lite

Fonction d’export `.tflite` intégrée :

- Utilisation possible sur **smartphones** (Android / iOS)
- Intégration dans des **microcontrôleurs** (Raspberry Pi, etc.)
- **Exécution rapide** et **hors ligne**

---

## 🖼️ Exemple d’usage pédagogique

> Un enseignant demande aux élèves de prendre des photos de fruits :  
> Chaque élève place ses photos dans un dossier par classe (pommes, bananes, oranges)  
> AI Creator entraîne le modèle  
> Les élèves testent leurs images avec des prédictions immédiates 🎯

---

## 💡 Pourquoi utiliser AI Creator ?

| Public        | Bénéfices |
|---------------|-----------|
| **Débutants** | Découverte visuelle du machine learning |
| **Étudiants** | Mini-projets IA locaux sans cloud |
| **Développeurs** | Prototypage rapide d’un classifieur personnalisé |
| **Écoles** | Introduction ludique et pédagogique à l’IA |

---

## 🔒 Vie privée

✅ Tout l’apprentissage est effectué **localement**, sans envoi de données sur Internet.  
✅ Aucune image n’est stockée ou partagée hors de l’ordinateur de l’utilisateur.

---

## 🧱 Évolutivité – pistes d’amélioration

- [x] Visualisation des courbes d'entraînement
- [x] Ajout de métriques (précision, rappel, F1)
- [x] Historique d’entraînement
- [x] Sauvegarde/chargement de modèles
- [x] Support de la webcam pour la prédiction en direct
- [x] Génération de fichier `.exe` pour Windows


---


## 📸 Fonctionnalités principales

| Fonction | Description |
|----------|-------------|
| 📂 Chargement de données | Choisissez un dossier avec des sous-dossiers d'images pour chaque classe |
| ➕ Ajout de classes | Créez de nouvelles classes directement dans l'interface et ajoutez des images |
| ⚙️ Paramétrage simple | Modifiez les dimensions des images, le nombre d’époques, la taille des batchs, etc. |
| 🧠 Entraînement de l’IA | Un modèle CNN est automatiquement compilé et entraîné sur vos données |
| 🔍 Test d’une image | Testez une image pour voir à quelle classe elle est associée |
| 💾 Sauvegarde du modèle | Le modèle est sauvegardé automatiquement à la fin de l'entraînement |
| 📤 Export TensorFlow Lite | Exportez votre modèle en `.tflite` pour un usage embarqué ou mobile |

---

## 🗂️ Organisation des données (dataset)

L'application s'appuie sur une structure simple : un dossier principal contenant un sous-dossier par classe, chacun rempli d'images. Exemple :

```
/mon_dataset/
├── chats/
│   ├── chat1.jpg
│   ├── chat2.jpg
├── chiens/
│   ├── chien1.jpg
│   ├── chien2.jpg
├── oiseaux/
│   ├── oiseau1.jpg
```

> 🔁 Vous pouvez ajouter de nouvelles classes à tout moment via le bouton **“Ajouter une classe”**.

---

## 🧰 Technologies utilisées

- **Python 3.10+** *(compatible GPU)*
- **TensorFlow** — entraînement du modèle IA
- **Keras** — architecture du modèle CNN
- **Tkinter** — interface graphique (GUI)
- **Pillow (PIL)** — manipulation d’images
- **NumPy** — traitement des tableaux d’images

---

## ⚙️ Installation (Windows)

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/ai-creator.git
cd ai-creator
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv env
env\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install tensorflow pillow numpy
```

*ou avec requirements.txt :*

```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer l’application

```bash
python Ai_creator_V6.py
```

L'interface graphique s'ouvre automatiquement.

---

## 🔍 Utilisation étape par étape

1. **Choisir un dossier d'images**
2. **Ajouter des classes (facultatif)**
3. **Configurer les paramètres d'entraînement**
4. **Lancer l'entraînement**
5. **Tester une image**
6. **Exporter en TFLite**

---

## 💻 Compatibilité GPU (Optionnel)

TensorFlow utilisera **automatiquement le GPU** si :
- Carte NVIDIA compatible CUDA
- CUDA Toolkit 11.8 + cuDNN 8.6 installés
- Drivers à jour

### Tester la présence du GPU :

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 📦 Export du modèle

- `model/model.h5` — modèle complet (Keras)
- `model.tflite` — modèle optimisé pour mobile (optionnel)

---

## 🐞 Dépannage courant

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | Installer avec `pip install nom_du_module` |
| L’environnement ne s’active pas | Utiliser `env\Scripts\activate.bat` |
| Entraînement lent | Vérifier l'utilisation du GPU ou réduire la taille des images |

---

## 📁 Fichier requirements.txt

```txt
tensorflow>=2.15
numpy>=1.25
pillow>=10.0
```

---

## 👨‍💻 Contribuer

Les contributions sont les bienvenues !  
Merci de soumettre une _issue_ ou une _pull request_ pour toute suggestion ou amélioration.

1. Forkez le dépôt
2. Créez une branche : `git checkout -b feature/ma-fonction`
3. Commitez : `git commit -am "Ajoute ma fonction"`
4. Poussez : `git push origin feature/ma-fonction`
5. Ouvrez une **Pull Request**

---

## 📄 Licence

Projet sous licence apache 2.0.

---

## 🤖 Auteur

**Créateur :** [Esteban Debordes]  
Contact : [esteban.pro32214@gmail.com]

