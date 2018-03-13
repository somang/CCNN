from keras.models import Sequential
from keras.layers import Dense, Input
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

# split input(X) and output(Y)
X = dataset[:,0:4]
Y = dataset[:,4:] #6, 7, 8, 9, 10

##### multivariate linear regression
#mlm = linear_model.LinearRegression()
#model = mlm.fit(X,Y)
#print(model.coef_)




# normalization
#sc = StandardScaler()
#X = sc.fit_transform(X)
#Y = sc.fit_transform(Y)

#print(Y)
#min_max_scaler = preprocessing.MinMaxScaler()
#Y = min_max_scaler.fit_transform(Y)
#print(Y)

#create model

model = Sequential()
model.add(Dense(units=64, input_dim=4, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(5))
#model.add(Dense(6, activation='sigmoid', kernel_initializer='normal'))

#inp_layer = Input(shape=(4,))
#out_layer = Dense(5, activation='relu')(inp_layer)
#model = Model(inputs=inp_layer, outputs=out_layer)

#compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
#fit the model
hist = model.fit(X, Y, epochs=50000, batch_size=10)
w = model.get_weights()
#save the model
model.save('model.h5') #creates a hdf5 file
end = time.time()
print(end-start)

# draw the loss graph
plt.plot(hist.history['loss'])
plt.ylim(0.0, 100)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#load model
#model = load_model('model.h5')
# plot model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# status, delay, wpm, similarity, number of errors
# speed, delay, missing words, grammar errors, verbatim
#predict using the model
p_input = np.array(
  [
    [0,120,1,0],  # 10,10,10,10,10
    [10000,120,1,0], # 10,0,10,10,10
    [7060,57.142,0.7292,0], # 6,0,0,10,0

    [2669,337.61,0.878,0], # 3,4,4,10,3
    [7271,814.15,0.689,1], # 1,1,3,10,0
    [4480,305.41,0.804,0], # 3,2,10,10,3
  ]
)
exp_output = np.array(
  [
    [10,10,10,10,10],
    [10,0,10,10,10],
    [10,0,10,10,10],
    [6,0,0,10,0],

    [3,4,4,10,3],
    [1,1,3,10,0],
    [3,2,10,10,3],
  ]
)
# stat linear regression
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