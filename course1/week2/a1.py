import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt



class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('acc')>0.99):
			print("accuracy reached 99% so cancelling training...")
			self.model.stop_training = True



callbacks = myCallback()
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(units=512, activation='relu'),
	keras.layers.Dense(units=10, activation='softmax')
	])



model.compile(optimizer='adam', 
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(train_images, 
		  train_labels, 
		  epochs=15, 
		  callbacks=[callbacks])

results = model.evaluate(test_images, test_labels)
print(results)