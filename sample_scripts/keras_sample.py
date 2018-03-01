from keras.models import Sequential
from keras.layers import Dense
import numpy

#fix random seed for reproducibility
#numpy.random.seed(7)


# 1 load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# 2 create model
model = Sequential()

# it defines the input layer having 8 inputs
# it defines a hidden layer with 12 neurons, connected to the input
# using relu activation function
# It initializes all weights using a sample of uniform random numbers.
model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3 compile
# binary classification -> loss=binary crossentropy
# adam optimizeer -> stochastic optimization
# Finally, because it is a classification problem, 
# we will collect and report the classification accuracy as the metric.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4 fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# 5 evaluate
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
print(X)
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)