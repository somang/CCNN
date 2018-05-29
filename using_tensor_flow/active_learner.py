import sys
import keras
import numpy as np
from keras.datasets import mnist, fashion_mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner

import matplotlib.pyplot as plt

def make_array(user_answer):
  tmp = []
  for i in range(10):
    if i == int(user_answer):
      tmp.append(1)
    else:
      tmp.append(0)
  return np.asarray([tmp])

# build function for the Keras' scikit-learn API
def create_keras_model():
  """
  This function compiles and returns a Keras model.
  Should be passed to KerasClassifier in the Keras scikit-learn API.
  """
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

  return model

# create the classifier
classifier = KerasClassifier(create_keras_model)

"""
Data wrangling
1. Reading data from Keras
2. Assembling initial training data for ActiveLearner
3. Generating the pool
"""

# read training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# assemble initial data
n_initial = 100
initial_idx = np.random.choice(
  range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

"""
Training the ActiveLearner
"""
# initialize ActiveLearner
learner = ActiveLearner(
  estimator = classifier,
  X_training = X_initial, 
  y_training = y_initial,
  verbose = 1)

# the active learning loop
n_queries = 2
for idx in range(n_queries):
  query_idx, query_instance = learner.query(X_pool, n_instances=1, verbose=0)
  tmp_index = query_idx[0]
  img = X_pool[tmp_index].reshape((28,28))
  classname = y_pool[tmp_index]
  plt.imshow(img)
  plt.title(classname)
  plt.show()
  print(y_pool[query_idx])
  print(y_pool[query_idx].shape)
  user_answer = input("What is this? [0-9]: ")
  user_answer = make_array(user_answer)

  learner.teach(
    X = X_pool[query_idx], 
    y = user_answer, 
    only_new = True, 
    verbose = 1)

  # remove queried instance from pool
  X_pool = np.delete(X_pool, query_idx, axis=0)
  y_pool = np.delete(y_pool, query_idx, axis=0)

# the final accuracy score
print(learner.score(X_test, y_test, verbose=1))