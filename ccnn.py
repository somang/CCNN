#from keras.models import Sequential
#from keras.layers import Dense
import numpy as np
from time import time
#from keras.callbacks import TensorBoard
#from keras.utils import plot_model

# setup data
  # import csv data and create an matrix
  # 0-3 : negative
  # 4-6 : no effect
  # 7-10 : positive
dataset = np.loadtxt("filtered_data_numbers.csv", delimiter=",")
print(len(dataset))

# split input(X) and output(Y)
Y = dataset[:,:]
print(Y[0])
#print(Y)
inp = [3319, 103.7037, 1, 0.97, 0]
X = np.asarray(inp)

'''
#create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
#model.add(Dense(1, activation='sigmoid'))

#compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# for visualization
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#fit the model
history = model.fit(X, Y, epochs=350, batch_size=10)
'''
'''
one epoch = one forward pass and one backward pass of 
            all the training examples
batch size = the number of training examples in one 
            forward/backward pass. The higher the batch size,
            the more memory space you'll need.
'''

# plot model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


'''
#predict using the model
p_input = np.array(
  [
    [3,126,88,41,235,39.3,0.704,27],
    [3,171,72,33,135,33.3,0.199,24]
  ]
  ) # should be 0,1
#print(p_input.shape)

prediction = model.predict(p_input)
print(prediction)
#print(round(prediction[0][0]))
'''