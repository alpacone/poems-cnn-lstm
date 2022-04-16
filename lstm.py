# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

trainingSet = pd.read_json('/content/drive/MyDrive/Colab Notebooks/training.json')
testingSet = pd.read_json('/content/drive/MyDrive/Colab Notebooks/testing.json')
df = pd.concat([trainingSet, testingSet])

df.head()

df.output.value_counts()

vocab_size = 14709
word_size = 64

model = keras.models.Sequential([
  layers.Embedding(vocab_size, word_size, mask_zero=True),
  layers.Bidirectional(layers.LSTM(word_size, return_sequences=True)),
  layers.Bidirectional(layers.LSTM(word_size//2)),
  layers.Dense(word_size, activation='relu'),
  layers.Dense(10)
])

model.summary()

df['syllables'] = df.input.apply(lambda x: len(x))
max_syllables = df.syllables.max()
total_posts = df.input.count()
print('max_syllables', max_syllables)
print('total_posts', total_posts)

X = keras.preprocessing.sequence.pad_sequences(df.input, padding='post')
Y = pd.get_dummies(df.output).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 28112021)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

epochs = 16
batch_size = 4

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

model.evaluate(X_test, Y_test, batch_size=batch_size)

tfjs.converters.save_keras_model(model, "/content/drive/MyDrive/Colab Notebooks/lstm-model.json")