import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import joblib
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf


class Patches(layers.Layer):
	def __init__(self, patch_size, **kwargs):
		super(Patches, self).__init__(**kwargs)
		self.patch_size = patch_size

	def call(self, images):
		batch_size = tf.shape(images)[0]
		patches = tf.image.extract_patches(
			images=images,
			sizes=[1, self.patch_size, self.patch_size, 1],
			strides=[1, self.patch_size, self.patch_size, 1],
			rates=[1, 1, 1, 1],
			padding="VALID",
		)
		patch_dims = patches.shape[-1]
		return tf.reshape(patches, [batch_size, -1, patch_dims])

	def get_config(self):
		config = super(Patches, self).get_config()
		config.update({"patch_size": self.patch_size})
		return config


class PatchEncoder(layers.Layer):
	def __init__(self, num_patches, projection_dim, **kwargs):
		super(PatchEncoder, self).__init__(**kwargs)
		self.num_patches = num_patches
		self.projection_dim = projection_dim
		self.projection = layers.Dense(units=projection_dim)
		self.position_embedding = layers.Embedding(
			input_dim=num_patches, output_dim=projection_dim
		)

	def call(self, patch):
		positions = tf.range(start=0, limit=self.num_patches, delta=1)
		encoded = self.projection(patch) + self.position_embedding(positions)
		return encoded

	def get_config(self):
		config = super(PatchEncoder, self).get_config()
		config.update({
			"num_patches": self.num_patches,
			"projection_dim": self.projection_dim
		})
		return config


class DigitRecognitionApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Handwritten Digit Recognition")
		self.root.geometry("600x400")

		self.image_path = None
		self.model_path = None
		self.model = None
		self.model_type = None
		self.expected_features = None
		self.h5_input_shape = None

		self.label_title = tk.Label(root, text="Handwritten Digit Recognition", font=("Arial", 16))
		self.label_title.pack(pady=10)

		self.btn_load_image = tk.Button(root, text="Load Image", command=self.load_image)
		self.btn_load_image.pack(pady=5)

		self.btn_load_model = tk.Button(root, text="Load Model", command=self.load_model)
		self.btn_load_model.pack(pady=5)

		self.btn_predict = tk.Button(root, text="Predict Digit", command=self.predict_digit)
		self.btn_predict.pack(pady=5)

		self.label_image = tk.Label(root)
		self.label_image.pack(pady=10)

		self.label_result = tk.Label(root, text="Prediction: None", font=("Arial", 12))
		self.label_result.pack(pady=10)

	def load_image(self):
		"""Load and display an image file."""
		file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
		if file_path:
			self.image_path = file_path

			img = Image.open(file_path)
			img = img.resize((100, 100))
			img_tk = ImageTk.PhotoImage(img)
			self.label_image.config(image=img_tk)
			self.label_image.image = img_tk
			self.label_result.config(text="Prediction: None")

	def load_model(self):
		"""Load a .pkl or .h5 model file."""
		file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl *.h5")])
		if file_path:
			self.model_path = file_path
			try:
				if file_path.endswith('.pkl'):
					self.model = joblib.load(file_path)
					self.model_type = 'pkl'

					if hasattr(self.model, 'coef_'):
						self.expected_features = self.model.coef_.shape[1]
					elif hasattr(self.model, 'theta_'):
						self.expected_features = self.model.theta_.shape[1]
					elif hasattr(self.model, 'n_features_in_'):
						self.expected_features = self.model.n_features_in_
					else:
						raise AttributeError("Unsupported scikit-learn model type: Cannot determine expected features.")
					self.h5_input_shape = None
				elif file_path.endswith('.h5'):

					custom_objects = {'Patches': Patches, 'PatchEncoder': PatchEncoder}
					self.model = keras.models.load_model(file_path, custom_objects=custom_objects)
					self.model_type = 'h5'
					self.expected_features = None

					self.h5_input_shape = self.model.input_shape
				messagebox.showinfo("Success", "Model loaded successfully!")
			except Exception as e:
				messagebox.showerror("Error", f"Failed to load model: {str(e)}")
				self.model = None
				self.model_type = None
				self.expected_features = None
				self.h5_input_shape = None

	def preprocess_image(self):
		if not self.image_path:
			messagebox.showerror("Error", "Please load an image first!")
			return None

		img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

		if self.model_type == 'pkl':
			if self.expected_features == 784:

				img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
			elif self.expected_features == 64:

				img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
			else:
				messagebox.showerror("Error", f"Unsupported number of features: {self.expected_features}")
				return None
		elif self.model_type == 'h5':
			if self.h5_input_shape is None:
				messagebox.showerror("Error", "Cannot determine input shape for .h5 model.")
				return None

			expected_shape = self.h5_input_shape[1:]
			if len(expected_shape) == 1:
				features = expected_shape[0]
				if features == 784:

					img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
					img = img.reshape(1, -1)
				else:
					messagebox.showerror("Error", f"Unsupported number of features for .h5 model: {features}")
					return None
			elif len(expected_shape) == 2:
				height, width = expected_shape
				img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
				img = img.reshape(1, height, width)
			elif len(expected_shape) == 3:
				height, width, channels = expected_shape
				if channels != 1:
					messagebox.showerror("Error", f"Unsupported number of channels: {channels}. Expected 1.")
					return None
				img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
				img = img.reshape(1, height, width, channels)
			else:
				messagebox.showerror("Error", f"Unsupported input shape for .h5 model: {self.h5_input_shape}")
				return None

		img = img.astype('float32') / 255.0

		if self.model_type == 'pkl':
			img = img.reshape(1, -1)
		return img

	def predict_digit(self):
		if not self.model:
			messagebox.showerror("Error", "Please load a model first!")
			return
		if not self.image_path:
			messagebox.showerror("Error", "Please load an image first!")
			return
		img = self.preprocess_image()
		if img is not None:
			try:
				if self.model_type == 'pkl':
					prediction = self.model.predict(img)
					digit = int(prediction[0])
				elif self.model_type == 'h5':
					prediction = self.model.predict(img, verbose=0)
					digit = np.argmax(prediction, axis=1)[0]
				self.label_result.config(text=f"Prediction: {digit}")
			except Exception as e:
				messagebox.showerror("Error", f"Prediction failed: {str(e)}")


if __name__ == "__main__":
	root = tk.Tk()
	app = DigitRecognitionApp(root)
	root.mainloop()
