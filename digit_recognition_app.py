import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import joblib
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import traceback


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
		self.model = None
		self.model_type = None
		self.expected_features = None
		self.h5_input_shape = None
		self.invert_image = tk.BooleanVar(value=False)

		self.model_names = [
			'KNeighbors_EMNIST_digits', 'LogisticRegression_EMNIST_digits',
			'GaussianNB_EMNIST_digits', 'DecisionTree_EMNIST_digits',
			'DNN_EMNIST_digits', 'SVM_model', 'LogisticRegression_model',
			'KNeighbors_model', 'GaussianNB_model', 'DecisionTree_model',
			'vit_mnist_model', 'KNeighbors_MNIST_model', 'dnn_mnist_model',
			'LogisticRegression_MNIST_model', 'GaussianNB_MNIST_model',
			'DecisionTree_MNIST_model'
		]

		self.model_paths = {
			'KNeighbors_EMNIST_digits': 'KNeighbors_EMNIST_digits.pkl',
			'LogisticRegression_EMNIST_digits': 'LogisticRegression_EMNIST_digits.pkl',
			'GaussianNB_EMNIST_digits': 'GaussianNB_EMNIST_digits.pkl',
			'DecisionTree_EMNIST_digits': 'DecisionTree_EMNIST_digits.pkl',
			'DNN_EMNIST_digits': 'DNN_EMNIST_digits.h5',
			'SVM_model': 'SVM_model.pkl',
			'LogisticRegression_model': 'LogisticRegression_model.pkl',
			'KNeighbors_model': 'KNeighbors_model.pkl',
			'GaussianNB_model': 'GaussianNB_model.pkl',
			'DecisionTree_model': 'DecisionTree_model.pkl',
			'vit_mnist_model': 'vit_mnist_model.h5',
			'KNeighbors_MNIST_model': 'KNeighbors_MNIST_model.pkl',
			'dnn_mnist_model': 'dnn_mnist_model.h5',
			'LogisticRegression_MNIST_model': 'LogisticRegression_MNIST_model.pkl',
			'GaussianNB_MNIST_model': 'GaussianNB_MNIST_model.pkl',
			'DecisionTree_MNIST_model': 'DecisionTree_MNIST_model.pkl'
		}

		self.label_title = tk.Label(root, text="Handwritten Digit Recognition", font=("Arial", 16))
		self.label_title.pack(pady=10)

		self.btn_load_image = tk.Button(root, text="Load Image", command=self.load_image)
		self.btn_load_image.pack(pady=5)

		self.label_model = tk.Label(root, text="Select Model:", font=("Arial", 12))
		self.label_model.pack(pady=5)

		self.model_var = tk.StringVar()
		self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, values=self.model_names, state="readonly")
		self.model_dropdown.bind("<<ComboboxSelected>>", self.load_selected_model)
		self.model_dropdown.pack(pady=5)

		self.invert_checkbox = tk.Checkbutton(root, text="Invert Image (for dark digits on white background)",
											  variable=self.invert_image)
		self.invert_checkbox.pack(pady=5)

		self.btn_predict = tk.Button(root, text="Predict Digit", command=self.predict_digit, state="disabled")
		self.btn_predict.pack(pady=5)

		self.label_image = tk.Label(root)
		self.label_image.pack(pady=10)

		self.label_result = tk.Label(root, text="Prediction: None", font=("Arial", 12))
		self.label_result.pack(pady=10)

	def check_predict_button_state(self):
		"""Enable predict button only if both image and model are selected."""
		if self.image_path and self.model:
			self.btn_predict.config(state="normal")
		else:
			self.btn_predict.config(state="disabled")

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
			self.check_predict_button_state()

	def load_selected_model(self, event=None):
		"""Load the selected model from the dropdown."""
		selected_model = self.model_var.get()
		if not selected_model:
			return

		file_path = self.model_paths.get(selected_model)
		if not file_path:
			messagebox.showerror("Error", f"Model path for {selected_model} not found!")
			return

		if not os.path.exists(file_path):
			messagebox.showerror("Error", f"Model file not found at: {file_path}")
			return

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
				messagebox.showinfo("Success", f"Model {selected_model} (.pkl) loaded successfully!")

			elif file_path.endswith('.h5'):
				custom_objects = {'Patches': Patches, 'PatchEncoder': PatchEncoder}
				try:
					self.model = keras.models.load_model(file_path, custom_objects=custom_objects, compile=False)
					self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
					self.model_type = 'h5'
					self.expected_features = None
					self.h5_input_shape = self.model.input_shape
					messagebox.showinfo("Success", f"Model {selected_model} (.h5) loaded successfully!")
				except Exception as e:
					error_details = traceback.format_exc()
					messagebox.showerror("Error",
										 f"Failed to load .h5 model {selected_model}: {str(e)}\nDetails: {error_details}")
					self.model = None
					self.model_type = None
					self.h5_input_shape = None
					self.check_predict_button_state()
					return
			self.check_predict_button_state()
		except Exception as e:
			error_details = traceback.format_exc()
			messagebox.showerror("Error", f"Failed to load model {selected_model}: {str(e)}\nDetails: {error_details}")
			self.model = None
			self.model_type = None
			self.expected_features = None
			self.h5_input_shape = None
			self.check_predict_button_state()

	def center_image(self, img):
		"""Center the digit in the image based on its center of mass."""

		moments = cv2.moments(img)
		if moments['m00'] == 0:
			return img
		cx = int(moments['m10'] / moments['m00'])
		cy = int(moments['m01'] / moments['m00'])

		rows, cols = img.shape
		shift_x = cols // 2 - cx
		shift_y = rows // 2 - cy

		M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
		centered_img = cv2.warpAffine(img, M, (cols, rows), borderValue=0)
		return centered_img

	def preprocess_image(self):
		if not self.image_path:
			messagebox.showerror("Error", "Please load an image first!")
			return None

		img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			messagebox.showerror("Error", "Failed to load image!")
			return None

		img = cv2.GaussianBlur(img, (3, 3), 0)

		img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

		img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

		img_bin = self.center_image(img_bin)

		img = img_bin.astype('float32') / 255.0

		if self.invert_image.get():
			img = 1.0 - img

		if self.model_type == 'pkl':
			if self.expected_features == 784:
				img = img.reshape(1, -1)
			elif self.expected_features == 64:
				img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
				img = img.reshape(1, -1)
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
					img = img.reshape(1, -1)
				else:
					messagebox.showerror("Error", f"Unsupported number of features for .h5 model: {features}")
					return None
			elif len(expected_shape) == 2:
				height, width = expected_shape
				if height != 28 or width != 28:
					messagebox.showerror("Error", f"Unsupported input shape for .h5 model: {expected_shape}")
					return None
				img = img.reshape(1, height, width)
			elif len(expected_shape) == 3:
				height, width, channels = expected_shape
				if height != 28 or width != 28 or channels != 1:
					messagebox.showerror("Error", f"Unsupported input shape for .h5 model: {expected_shape}")
					return None
				img = img.reshape(1, height, width, channels)
			else:
				messagebox.showerror("Error", f"Unsupported input shape for .h5 model: {self.h5_input_shape}")
				return None

		return img

	def predict_digit(self):
		if not self.model or not self.image_path:
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
