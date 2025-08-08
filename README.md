# 🧠 AI Creator - Teachable Image Classifier (v6)

**AI Creator** is a Python desktop application with a graphical interface (Tkinter) that allows anyone to train an image classification **AI model visually and intuitively**, without writing a single line of code.

Inspired by Google’s [Teachable Machine](https://teachablemachine.withgoogle.com/), this local version ensures privacy, customization, and flexibility while remaining simple to use.

---

## 🎯 Project Goal

**AI Creator** aims to democratize artificial intelligence by enabling any user, even without programming skills, to train an image classification model using their own data in just a few clicks.

This project allows:

- **Beginners** to discover machine learning visually,
- **Teachers** to offer interactive classroom activities,
- **Developers** to quickly prototype a **local** computer vision model.

---

## 🖥️ User Interface

The application features an intuitive interface built with **Tkinter**, divided into two sections:

### 🔹 Left Panel – Class Management

- Display of detected classes
- “Add Class” button (manual image import)

### 🔹 Right Panel – Model Controls

- Image folder selection
- Hyperparameter settings: size, batch, epochs, patience
- Progress bars + live training log
- **“Test an Image”** button
- **“Export to TensorFlow Lite”** button

---

## 🧠 Technical Details

The model is a **simple CNN** implemented with **Keras** and **TensorFlow**:

**Architecture:**
```
Input → Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(128) → Dense(n_classes, softmax)
```

- **Activation functions:** ReLU (hidden layers), Softmax (output)
- **Loss:** `categorical_crossentropy`
- **Optimizer:** `Adam`
- **Metric:** `accuracy`
- **EarlyStopping** on validation loss

---

## ⚙️ Customization

Users can configure:

- Image size (width x height)
- Batch size
- Number of epochs
- Patience threshold (EarlyStopping)

---

## 🧪 Test Function

Once the model is trained, users can:

- Load an image
- Instantly get a prediction
- See the name of the predicted class

---

## 📤 TensorFlow Lite Export

Built-in `.tflite` export feature:

- Usable on **smartphones** (Android / iOS)
- Integration with **microcontrollers** (Raspberry Pi, etc.)
- **Fast** and **offline execution**

---

## 🖼️ Educational Use Case Example

> A teacher asks students to take photos of fruits:  
> Each student places their photos in a folder per class (apples, bananas, oranges)  
> AI Creator trains the model  
> Students test their images with immediate predictions 🎯

---

## 💡 Why Use AI Creator?

| Audience        | Benefits |
|-----------------|----------|
| **Beginners**   | Visual discovery of machine learning |
| **Students**    | Local AI mini-projects without the cloud |
| **Developers**  | Rapid prototyping of a custom classifier |
| **Schools**     | Fun and educational introduction to AI |

---

## 🔒 Privacy

✅ All training is done **locally**, with no data sent over the Internet.  
✅ No images are stored or shared outside the user’s computer.

---

## 🧱 Scalability – Improvement Ideas

- [x] Training curve visualization
- [x] Added metrics (precision, recall, F1)
- [x] Training history
- [x] Model save/load support
- [x] Webcam support for live prediction
- [x] `.exe` generation for Windows

---

## 📸 Key Features

| Feature              | Description |
|----------------------|-------------|
| 📂 Data loading       | Choose a folder with subfolders of images for each class |
| ➕ Add classes        | Create new classes directly in the UI and add images |
| ⚙️ Simple setup       | Modify image size, epochs, batch size, etc. |
| 🧠 AI training        | A CNN model is automatically compiled and trained on your data |
| 🔍 Image testing      | Test an image to see which class it belongs to |
| 💾 Model saving       | The model is automatically saved at the end of training |
| 📤 TensorFlow Lite export | Export your model to `.tflite` for embedded or mobile use |

---

## 🗂️ Data Organization (Dataset)

The application uses a simple folder structure: a main folder with one subfolder per class, each filled with images. Example:

```
/my_dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
├── birds/
│   ├── bird1.jpg
```

> 🔁 You can add new classes anytime using the **“Add Class”** button.

---

## 🧰 Technologies Used

- **Python 3.10+** *(GPU compatible)*
- **TensorFlow** — AI model training
- **Keras** — CNN model architecture
- **Tkinter** — Graphical user interface
- **Pillow (PIL)** — Image processing
- **NumPy** — Array/image processing

---

## ⚙️ Installation (Windows)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-creator.git
cd ai-creator
```

### 2. Create and activate a virtual environment

```bash
python -m venv env
env\Scriptsctivate
```

### 3. Install dependencies

```bash
pip install tensorflow pillow numpy
```

*Or with requirements.txt:*

```bash
pip install -r requirements.txt
```

---

## ▶️ Launch the Application

```bash
python Ai_creator_V6.py
```

The graphical interface will open automatically.

---

## 🔍 Step-by-Step Usage

1. **Choose an image folder**
2. **Add classes (optional)**
3. **Set training parameters**
4. **Start training**
5. **Test an image**
6. **Export to TFLite**

---

## 💻 GPU Compatibility (Optional)

TensorFlow will **automatically use the GPU** if:
- NVIDIA card with CUDA support
- CUDA Toolkit 11.8 + cuDNN 8.6 installed
- Drivers up to date

### Test for GPU presence:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 📦 Model Export

- `model/model.h5` — full Keras model
- `model.tflite` — mobile-optimized model (optional)

---

## 🐞 Common Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Install with `pip install module_name` |
| Environment doesn't activate | Use `env\Scriptsctivate.bat` |
| Slow training | Check GPU usage or reduce image size |

---

## 📁 requirements.txt File

```txt
tensorflow>=2.15
numpy>=1.25
pillow>=10.0
```

---

## 👨‍💻 Contributing

Contributions are welcome!  
Please submit an _issue_ or _pull request_ for any suggestion or improvement.

1. Fork the repository  
2. Create a branch: `git checkout -b feature/my-feature`  
3. Commit: `git commit -am "Add my feature"`  
4. Push: `git push origin feature/my-feature`  
5. Open a **Pull Request**

---

## 📄 License

Project licensed under Apache 2.0.

---

## 🤖 Author

**Creator:** [Esteban Debordes]  
Contact: [esteban.pro32214@gmail.com]
