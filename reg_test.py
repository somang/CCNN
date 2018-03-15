from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.models import load_model

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import time

x_train = np.random.rand(9000)
y_train = x_train**4+x_train**3-x_train
x_train = x_train.reshape(len(x_train),1)

x_test = np.linspace(0,1,100)
y_test = x_test**4+x_test**3-x_test
x_test = x_test.reshape(len(x_test),1)


model = Sequential()
model.add(Dense(units=20, input_dim=1, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error',
              optimizer='sgd', 
              metrics=['accuracy']
              )

hist = model.fit(x_train, y_train, 
                epochs=40, batch_size=50,
                verbose=1, 
                validation_data=(x_test, y_test)
                )

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)

predictions = model.predict(x_test, batch_size=1)

x_test = x_test.reshape(-1)
plt.plot(x_test, y_test, c='b')
plt.plot(x_test, predictions, c='r')
plt.show()