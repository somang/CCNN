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
dataset = np.loadtxt("gen_dt_10000.csv", delimiter=",")
print(len(dataset))
t_index = round(len(dataset)*0.8)

# normalization
sc = StandardScaler()
X_data = sc.fit_transform(dataset[:,:1])
#X_data = dataset[:,:1]
Y_data = dataset[:,4:5]

# split input(X) and output(Y)
X_train, Y_train = X_data[:t_index,:], Y_data[:t_index,:]
X_test, Y_test = X_data[t_index:,:], Y_data[t_index:,:]

#create model
# using sequential
model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(1, activation='relu'))


# using functional API
#in_layer = Input(shape=(3,)) #number of columns
#hidden_1 = Dense(40, activation='relu', kernel_initializer='normal')(in_layer)
#out_layer = Dense(1, activation='relu')(in_layer)
#model = Model(inputs=in_layer, outputs=out_layer)

# summarize the model
model.summary()
# plot model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


#compile model
#model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

#fit the model
hist = model.fit(X_train, Y_train, 
                epochs=100, batch_size=500,
                verbose=1, validation_data=(X_test, Y_test)
                )
w = model.get_weights()
print(w)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


'''
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
'''

'''
# delay, wpm, similarity
# speed, delay, verbatim
#predict using the model
p_input = np.array(
  [
    [0,120,1],  # 10,10,10
    [10000,120,1], # 10,0,10
    [7060,57.142,0.7292], # 3,<4,3

    [2669,337.61,0.878], # 1, 3, 7
    [7271,814.15,0.689], # 1, 1, 4
    [4480,305.41,0.804], # 3, 2, 3
  ]
)

#scale
p_input = sc.transform(p_input)
prediction = model.predict(p_input)
print(prediction)
for i in prediction:
  print(list(map(lambda x: round(x), i)))
'''


'''
  unit = 
  optimizer = 
        adam:
        rmsprop:
        stochastic gradient descent:
  one epoch = one forward pass and one backward pass of 
              all the training examples
  batch size = the number of training examples in one 
              forward/backward pass. The higher the batch size,
              the more memory space you'll need.
'''



##### multivariate linear regression
mlm = linear_model.LinearRegression()
stat_model = mlm.fit(X_train, Y_train[:,0])
predictions = mlm.predict(X_test)
## plot the mlm
plt.scatter(Y_test, predictions)
plt.xlabel("Real Values")
plt.ylabel("Predictions")
plt.show()
# print the accuracy score
print("Score:", mlm.score(X_test, Y_test[:,0]))
#prediction = mlm.predict(p_input)
#print(prediction)
#for i in prediction:
#  print(list(map(lambda x: round(x), i)))
