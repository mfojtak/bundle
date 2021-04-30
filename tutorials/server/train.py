import tensorflow as tf
mnist = tf.keras.datasets.mnist
import os
import numpy as np
from typing import List
import typing
from PIL import Image
import io

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model_path = "/data/mnist_model"
if os.path.exists(model_path):
  model = tf.keras.models.load_model(model_path)
else:
  model.fit(x_train, y_train, epochs=5)
  model.save(model_path)
  model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

def predict(input: List[List[List[float]]]) -> List[List[float]]:
  """
  Predicts batch of digits represented as 2D vectors (28x28).
  """
  npa = np.array(input)
  return probability_model(npa).numpy().tolist()

def predict_image(image: typing.BinaryIO) -> List[float]:
  """
  Predicts one image represented as binary IO input (e.g. file)
  """
  return predict_images([image])[0]

def predict_images(images: List[typing.BinaryIO]) -> List[List[float]]:
  numpy_images = []
  for image in images:
    pil_image = Image.open(io.BytesIO(image.read()))
    numpy_image = np.array(pil_image)
    numpy_image = numpy_image / 255
    numpy_images.append(numpy_image)
  return predict(numpy_images)

#print(predict(x_test[:5].tolist()))
#file = open("sample.png", "rb")
#print(predict_images([file.read()]))

