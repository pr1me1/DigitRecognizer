import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import joblib
import tensorflow.keras as keras
from skimage.transform import resize


class DigitRecognitionApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Handwritten Digit Recognition")
		self.root.geometry("600x400")

		self.image_path = None
		self.models = {}
		self.model_types = {}
		self.expected_features = {}
		self.h5_input_shapes = {}
		self.invert_image = tk.BooleanVar(value=False)

		self.model_paths = {
			"LogisticRegression": "LogisticRegression.pkl",
			"DecisionTree": "DecisionTree.pkl",
			"KNeighbors": "KNeighbors.pkl",
			"GaussianNB": "GaussianNB.pkl",
			"CNN": "Cnn.h5"
		}

		self.load_models()

		self.label_title = tk.Label(root, text="Qo‘l bilan yozilgan raqamni aniqlash", font=("Arial", 16))
		self.label_title.pack(pady=10)

		self.btn_load_image = tk.Button(root, text="Rasm yuklash", command=self.load_image)
		self.btn_load_image.pack(pady=5)

		self.invert_checkbox = tk.Checkbutton(root, text="Rasmni teskari qilish (qora fonli oq raqamlar uchun)",
											  variable=self.invert_image)
		self.invert_checkbox.pack(pady=5)

		self.btn_predict = tk.Button(root, text="Raqamni aniqlash", command=self.predict_digit, state="disabled")
		self.btn_predict.pack(pady=5)

		self.label_image = tk.Label(root)
		self.label_image.pack(pady=10)

		self.label_result = tk.Label(root, text="Natija: Yo‘q", font=("Arial", 12), justify="left")
		self.label_result.pack(pady=10)

	def load_models(self):
		for model_name, file_path in self.model_paths.items():
			if not os.path.exists(file_path):
				messagebox.showwarning("Warning", f"Model file not found: {file_path}. Skipping {model_name}.")
				continue
			try:
				if model_name == "CNN":

					self.models[model_name] = keras.models.load_model(file_path, compile=False)
					self.models[model_name].compile(optimizer='adam', loss='sparse_categorical_crossentropy',
													metrics=['accuracy'])
					self.model_types[model_name] = 'h5'
					self.expected_features[model_name] = None
					self.h5_input_shapes[model_name] = self.models[model_name].input_shape
				else:

					self.models[model_name] = joblib.load(file_path)
					self.model_types[model_name] = 'pkl'
					if hasattr(self.models[model_name], 'coef_'):
						self.expected_features[model_name] = self.models[model_name].coef_.shape[1]
					elif hasattr(self.models[model_name], 'n_features_in_'):
						self.expected_features[model_name] = self.models[model_name].n_features_in_
					else:
						messagebox.showwarning("Warning", f"Unsupported scikit-learn model: {model_name}")
						continue
					self.h5_input_shapes[model_name] = None
				print(f"Loaded model: {model_name}")
			except Exception as e:
				messagebox.showwarning("Warning", f"Failed to load model {model_name}: {str(e)}")
				self.models.pop(model_name, None)

		if not self.models:
			messagebox.showerror("Error", "No models could be loaded. Please check model files.")
			self.btn_predict.config(state="disabled")

	def check_predict_button_state(self):
		if self.image_path and self.models:
			self.btn_predict.config(state="normal")
		else:
			self.btn_predict.config(state="disabled")

	def load_image(self):
		file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
		if file_path:
			self.image_path = file_path
			img = Image.open(file_path)
			img = img.resize((100, 100))
			img_tk = ImageTk.PhotoImage(img)
			self.label_image.config(image=img_tk)
			self.label_image.image = img_tk
			self.label_result.config(text="Natija: Yo‘q")
			self.check_predict_button_state()

	def preprocess_image(self):
		if not self.image_path:
			messagebox.showerror("Error", "Please load an image first!")
			return None

		img = Image.open(self.image_path)
		resized_image = img.resize((256, 256))
		grayscale_image = resized_image.convert('L')
		grayscale_array = np.array(grayscale_image)
		normalized_array = grayscale_array / 255.0
		sharpened_img = grayscale_image.filter(ImageFilter.SHARPEN)
		sharpened_array = np.array(sharpened_img) / 255.0

		img_for_sklearn = resize(sharpened_array, (32, 32), anti_aliasing=True)
		flattened_img = img_for_sklearn.flatten()

		img_for_cnn = img_for_sklearn.reshape(1, 32, 32, 1)

		if self.invert_image.get():
			flattened_img = 1.0 - flattened_img
			img_for_cnn = 1.0 - img_for_cnn

		return {"sklearn": flattened_img, "cnn": img_for_cnn}

	def predict_digit(self):
		if not self.models or not self.image_path:
			return

		processed_images = self.preprocess_image()
		if processed_images is None:
			return

		predictions = []
		for model_name, model in self.models.items():
			try:
				if self.model_types[model_name] == 'pkl':
					if self.expected_features.get(model_name) != 1024:
						messagebox.showerror("Error",
											 f"Model {model_name} expects {self.expected_features[model_name]} features, but got 1024.")
						continue
					prediction = model.predict([processed_images["sklearn"]])
					digit = int(prediction[0])
				elif self.model_types[model_name] == 'h5':
					if self.h5_input_shapes[model_name][1:3] != (32, 32):
						messagebox.showerror("Error",
											 f"Model {model_name} expects input shape {self.h5_input_shapes[model_name]}, but got (32, 32, 1).")
						continue
					prediction = model.predict(processed_images["cnn"], verbose=0)
					digit = np.argmax(prediction, axis=1)[0]
				predictions.append(f"{model_name}: {digit}")
			except Exception as e:
				predictions.append(f"{model_name}: Error ({str(e)})")

		result_text = "Natija:\n" + "\n".join(predictions) if predictions else "Natija: Hech qanday bashorat yo‘q"
		self.label_result.config(text=result_text)


if __name__ == "__main__":
	root = tk.Tk()
	app = DigitRecognitionApp(root)
	root.mainloop()
