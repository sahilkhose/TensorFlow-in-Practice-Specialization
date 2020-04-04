import tensorflow as tf
print(tf.__version__)
tf.enable_eager_execution()
import tensorflow_datasets as tfds 

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
	training_sentences.append(str(s.tonumpy()))
	training_labels.append(l.numpy())

for s, l in test_data:
	testing_sentences.append(str(s.tonumpy()))
	testing_labels.append(l.numpy())