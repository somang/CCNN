from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from time import time
from keras.callbacks import TensorBoard
from keras.utils import plot_model

from sklearn import preprocessing

# setup data
  # import csv data and create an matrix
  # 0-3 : negative
  # 4-6 : no effect
  # 7-10 : positive
dataset = np.loadtxt("sample.csv", delimiter=",")
print(len(dataset))

# split input(X) and output(Y)
X = dataset[:,:5]
Y = dataset[:,5:] #6, 7, 8, 9, 10, 11
# normalization
#print(Y)
#min_max_scaler = preprocessing.MinMaxScaler()
#Y = min_max_scaler.fit_transform(Y)
#print(Y)


#create model
model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='sigmoid', kernel_initializer='normal'))

#compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# for visualization
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#fit the model
history = model.fit(X, Y, epochs=50000, batch_size=12)



'''
one epoch = one forward pass and one backward pass of 
            all the training examples
batch size = the number of training examples in one 
            forward/backward pass. The higher the batch size,
            the more memory space you'll need.
'''

# plot model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# status, delay, wpm, similarity, number of errors
# speed, delay, missing words, grammer errors, speaker id, verbatim
#predict using the model
p_input = np.array(
  [
    [0,0,120,1,0],  # 10,10,10,10,10,10
    [1,10000,120,1,0],  # 10,0,10,10,10,10
    [0, 0, 703.7037, 0.3, 10], # should be [3,2,5,5,10,1]
    [0, 10000, 120, 1, 0] # 10,0,10,10,10,10
  ]
  ) 

prediction = model.predict(p_input)
print(prediction)

for i in prediction:
  print(list(map(lambda x: round(x), i)))