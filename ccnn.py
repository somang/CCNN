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


start = time.time()
# setup data
  # import csv data and create an matrix
dataset = np.loadtxt("generated_data.csv", delimiter=",")
print(len(dataset))
t_index = round(len(dataset)*0.8)

# normalization
sc = StandardScaler()
X_data = sc.fit_transform(dataset[:,:4])
#X_data = dataset[:,:4]
Y_data = dataset[:,4:]

# split input(X) and output(Y)
X_train, Y_train = X_data[:t_index,:], Y_data[:t_index,:]
X_test, Y_test = X_data[t_index:,:], Y_data[t_index:,:]

##### multivariate linear regression
#mlm = linear_model.LinearRegression()
#model = mlm.fit(X,Y)
#print(model.coef_)

#print(Y)
#min_max_scaler = preprocessing.MinMaxScaler()
#Y = min_max_scaler.fit_transform(Y)
#print(Y)

#create model

model = Sequential()
model.add(Dense(units=64, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(5, kernel_initializer='normal'))
model.summary()

'''
  [32, 12, 12, 5], 500 epochs = 60%
'''


#compile model
#model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])


#fit the model
hist = model.fit(X_train, Y_train, 
                epochs=1500, batch_size=250,
                verbose=1, validation_data=(X_test, Y_test)
                )
w = model.get_weights()
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#save the model
model.save('model.h5') #creates a hdf5 file
end = time.time()
print("TIME:", end-start)

#val loss graph
loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.show()
# draw the loss graph
acc = hist.history['acc']
val_acc = hist.history['val_acc']
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc', 'val_acc'])
plt.show()

#load model
#model = load_model('model.h5')
# plot model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# delay, wpm, similarity, number of errors
# speed, delay, missing words, grammar errors, verbatim
#predict using the model
p_input = np.array(
  [
    [0,120,1,0],  # 10,10,10,10,10
    [10000,120,1,0], # 10,0,10,10,10
    [7060,57.142,0.7292,0], # 3,0-3,0-10,0

    [2669,337.61,0.878,0], # 3,4,4,10,3
    [7271,814.15,0.689,1], # 1,1,3,10,0
    [4480,305.41,0.804,0], # 3,2,10,10,3
  ]
)

#scale
p_input = sc.transform(p_input)
prediction = model.predict(p_input)
print(prediction)
for i in prediction:
  print(list(map(lambda x: round(x), i)))



'''
  one epoch = one forward pass and one backward pass of 
              all the training examples
  batch size = the number of training examples in one 
              forward/backward pass. The higher the batch size,
              the more memory space you'll need.
'''