import numpy as np


## this script will make a ANN
# that is to train and predict the XOR logic
# using only numpy
def sigmoid(x, deriv=False):
  if deriv == True:
    return x*(1-x)
  return 1/(1+np.exp(-x))

#input
X = np.array([
  [0,0,1],
  [0,1,1],
  [1,0,1],
  [1,1,1]])
#output
Y = np.array([
  [0],[1],[1],[0]
  ])

#synapses, with random weight assigned to it
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# training step
for i in range(60000):
  l0 = X
  l1 = sigmoid(np.dot(l0, syn0))
  l2 = sigmoid(np.dot(l1, syn1))

  l2_error = Y - l2

  if i%10000 == 0:
    print("err:" + str(np.mean(np.abs(l2_error))))
  
  l2_delta = l2_error*sigmoid(l2, deriv=True)
  l1_error = l2_delta.dot(syn1.T)
  l1_delta = l1_error * sigmoid(l1, deriv=True)

  #update weights from backpropagation
  syn1 += l1.T.dot(l2_delta)
  syn0 += l0.T.dot(l1_delta)

print("output after training")
print(l2)