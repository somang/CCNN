from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import models
from keras import layers
from keras import optimizers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)



# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
#we reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

# decode the review; note that our indices were offset by 3
# because 0,1, and 2 are reserved indices for padding,
# start of sequence, and unknown.
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
#print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
  # create all-zero matrix of shape (len(sequences), dimension)
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1 # set speicfic indices of results[i] to 1s
  return results

#vectorize training and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
#print(x_train[0])
#print(train_labels[0], y_train[0])

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=100,
                    batch_size=128,
                    validation_data=(x_val, y_val))

history_dict = history.history
#print(history_dict.keys())


print(min(model.predict(x_test)))