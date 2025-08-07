import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from PIL import Image
import threading
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping
import time

DEFAULT_IMAGE_SIZE = (150, 150)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_PATIENCE = 3
MODEL_PATH = "model/model.h5"

class TrainingCallback(Callback):
    def __init__(self, app):
        self.app = app
        self.start_time = None
        self.epoch_times = []
        super().__init__()

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        avg_time_per_epoch = elapsed_time / (epoch + 1)
        estimated_total_time = avg_time_per_epoch * self.app.epochs
        remaining_time = estimated_total_time - elapsed_time

        def format_seconds(seconds):
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m {secs:02d}s"

        progress_percent = (epoch + 1) / self.app.epochs * 100
        self.app.progress_var.set(progress_percent)
        self.app.progress_bar.update()

        log_message = (
            f"\n\u00c9poque {epoch + 1}/{self.app.epochs} ({progress_percent:.1f}%)"
            f" - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}"
            f" - \u23f1 \u00c9coulé: {format_seconds(elapsed_time)}"
            f" | Estimé: {format_seconds(estimated_total_time)}"
            f" | Reste: {format_seconds(remaining_time)}"
        )
        self.app.log_text.insert(tk.END, log_message)
        self.app.log_text.yview(tk.END)

        if self.app.stop_training:
            self.model.stop_training = True
            self.app.status_label.config(text="⛔️ Entraînement arrêté.")

class TeachableApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Teachable Image Classifier")
        self.dataset_path = None
        self.model = None
        self.stop_training = False

        self.image_size = DEFAULT_IMAGE_SIZE
        self.batch_size = DEFAULT_BATCH_SIZE
        self.epochs = DEFAULT_EPOCHS
        self.patience = DEFAULT_PATIENCE

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.class_listbox = tk.Listbox(self.left_frame, height=20, width=40)
        self.class_listbox.pack(pady=10)

        self.btn_add_class = tk.Button(self.left_frame, text="Ajouter une classe (images)", command=self.add_class)
        self.btn_add_class.pack(pady=5)

        self.label = tk.Label(self.right_frame, text="Aucune donnée chargée.")
        self.label.pack(pady=10)

        self.btn_load = tk.Button(self.right_frame, text="1. Choisir dossier d'images", command=self.load_dataset)
        self.btn_load.pack(pady=5)

        self.param_frame = tk.LabelFrame(self.right_frame, text="Paramètres d'entraînement")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(self.param_frame, text="Largeur :").grid(row=0, column=0)
        self.entry_width = tk.Entry(self.param_frame, width=5)
        self.entry_width.insert(0, str(self.image_size[0]))
        self.entry_width.grid(row=0, column=1)

        tk.Label(self.param_frame, text="Hauteur :").grid(row=0, column=2)
        self.entry_height = tk.Entry(self.param_frame, width=5)
        self.entry_height.insert(0, str(self.image_size[1]))
        self.entry_height.grid(row=0, column=3)

        tk.Label(self.param_frame, text="Batch :").grid(row=1, column=0)
        self.entry_batch = tk.Entry(self.param_frame, width=5)
        self.entry_batch.insert(0, str(self.batch_size))
        self.entry_batch.grid(row=1, column=1)

        tk.Label(self.param_frame, text="Époques :").grid(row=1, column=2)
        self.entry_epochs = tk.Entry(self.param_frame, width=5)
        self.entry_epochs.insert(0, str(self.epochs))
        self.entry_epochs.grid(row=1, column=3)

        tk.Label(self.param_frame, text="Patience :").grid(row=2, column=0)
        self.entry_patience = tk.Entry(self.param_frame, width=5)
        self.entry_patience.insert(0, str(self.patience))
        self.entry_patience.grid(row=2, column=1)

        self.btn_update_params = tk.Button(self.param_frame, text="Mettre à jour", command=self.set_parameters)
        self.btn_update_params.grid(row=2, column=2, columnspan=2)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.right_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.status_label = tk.Label(self.right_frame, text="En attente.", font=("Helvetica", 12))
        self.status_label.pack(pady=5)

        self.btn_train = tk.Button(self.right_frame, text="2. Entraîner le modèle", command=self.start_training_thread)
        self.btn_train.pack(pady=5)

        self.btn_stop = tk.Button(self.right_frame, text="Arrêter l'entraînement", command=self.stop_training_callback)
        self.btn_stop.pack(pady=5)

        self.btn_test = tk.Button(self.right_frame, text="3. Tester une image", command=self.test_image)
        self.btn_test.pack(pady=5)

        self.btn_export_tflite = tk.Button(self.right_frame, text="4. Exporter en TensorFlow Lite", command=self.export_tflite)
        self.btn_export_tflite.pack(pady=5)

        self.log_text = tk.Text(self.right_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_label = tk.Label(self.right_frame, text="")
        self.result_label.pack(pady=10)

    def set_parameters(self):
        try:
            width = int(self.entry_width.get())
            height = int(self.entry_height.get())
            batch = int(self.entry_batch.get())
            epochs = int(self.entry_epochs.get())
            patience = int(self.entry_patience.get())

            self.image_size = (width, height)
            self.batch_size = batch
            self.epochs = epochs
            self.patience = patience

            messagebox.showinfo("OK", "Paramètres mis à jour.")
        except:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def stop_training_callback(self):
        self.stop_training = True
        self.status_label.config(text="⛔️ Entraînement interrompu.")

    def start_training_thread(self):
        self.stop_training = False
        self.status_label.config(text="⏳ En cours d'entraînement...")
        threading.Thread(target=self.train_model).start()

    def load_dataset(self):
        self.dataset_path = filedialog.askdirectory()
        if self.dataset_path:
            self.label.config(text=f"Dossier sélectionné : {self.dataset_path}")
            self.refresh_class_list()
        else:
            self.label.config(text="Aucun dossier sélectionné.")

    def refresh_class_list(self):
        self.class_listbox.delete(0, tk.END)
        if self.dataset_path:
            classes = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
            for cls in classes:
                count = len(os.listdir(os.path.join(self.dataset_path, cls)))
                self.class_listbox.insert(tk.END, f"{cls} ({count} images)")

    def add_class(self):
        if not self.dataset_path:
            messagebox.showwarning("Info", "Veuillez d'abord choisir un dossier d'images.")
            return

        new_class_name = tk.simpledialog.askstring("Nouvelle classe", "Nom de la nouvelle classe :")
        if not new_class_name:
            return

        target_dir = os.path.join(self.dataset_path, new_class_name)
        os.makedirs(target_dir, exist_ok=True)

        image_files = filedialog.askopenfilenames(title="Sélectionner des images pour la classe",
                                                  filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not image_files:
            return

        for img_path in image_files:
            try:
                filename = os.path.basename(img_path)
                new_path = os.path.join(target_dir, filename)
                with open(img_path, 'rb') as f_src:
                    with open(new_path, 'wb') as f_dst:
                        f_dst.write(f_src.read())
            except Exception as e:
                print(f"Erreur lors de la copie de {img_path} : {e}")

        self.refresh_class_list()
        messagebox.showinfo("Ajout terminé", f"{len(image_files)} images ajoutées à la classe « {new_class_name} »")

    def test_image(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Erreur", "Veuillez entraîner un modèle d'abord.")
            return

        if not self.model:
            self.model = load_model(MODEL_PATH)

        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not file_path:
            return

        img = Image.open(file_path).resize(self.image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        class_index = np.argmax(prediction[0])

        datagen = ImageDataGenerator(rescale=1./255)
        gen = datagen.flow_from_directory(self.dataset_path, target_size=self.image_size, batch_size=1, class_mode='categorical')
        class_labels = list(gen.class_indices.keys())

        result = class_labels[class_index]
        self.result_label.config(text=f"Prédiction : {result}")

    def export_tflite(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Erreur", "Veuillez entraîner un modèle d'abord.")
            return

        try:
            model = load_model(MODEL_PATH)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            save_path = filedialog.asksaveasfilename(defaultextension=".tflite",
                                                     filetypes=[("TFLite files", "*.tflite")])
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(tflite_model)
                messagebox.showinfo("Succès", f"Modèle exporté en TensorFlow Lite :\n{save_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de la conversion : {e}")

    def train_model(self):
        if not self.dataset_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier d'images d'abord.")
            return

        self.set_parameters()

        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_gen = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        num_classes = len(train_gen.class_indices)

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [EarlyStopping(patience=self.patience, restore_best_weights=True), TrainingCallback(self)]

        self.log_text.delete(1.0, tk.END)
        model.fit(train_gen, validation_data=val_gen, epochs=self.epochs, callbacks=callbacks)

        if not self.stop_training:
            self.status_label.config(text="✅ Entraînement terminé.")

        os.makedirs("model", exist_ok=True)
        model.save(MODEL_PATH)
        self.model = model
        messagebox.showinfo("Succès", "Modèle entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TeachableApp(root)
    root.mainloop()
