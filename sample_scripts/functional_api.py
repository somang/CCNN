from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

'''
#https://machinelearningmastery.com/keras-functional-api-deep-learning/
# sequential model:
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))

# functional model:
# providing a more flexible way for definining models
# specifically allows to define multiple input/output 
#
# 1. input
visible = Input(shape=(2,))
# 2. connecting layers
hidden = Dense(2)(visible) # connecting input to this hidden layer.
# 3. model creation
model = Model(inputs=visible, outputs=hidden)
'''

# multiplayer perceptron
# having 10 inputs, 3 hidden layers with 10, 20, 10 neurons
# and an output layer with 1 output.
# rectified linear activation functions are used
# binary classification

visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multiplayer_perceptron_graph.png')



'''
long short-term memory recurrent neural network 
for sequence classification.

The model expects 100 time steps of one feature as input. 
The model has a single LSTM hidden layer to extract features 
from the sequence, followed by a fully connected layer to 
interpret the LSTM output, 

followed by an output layer for making binary predictions.
'''
from keras.layers.recurrent import LSTM
visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
#summarize layers
print(model.summary())
#plot
