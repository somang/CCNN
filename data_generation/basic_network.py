import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
  if deriv == True:
    return x * (1 - x)
  return 1 / (1 + np.exp(-x))

# input dataset
x = np.array([
  [0,0,1],
  [0,1,1],
  [1,0,1],
  [1,1,1]
])

# output dataset
y = np.array([[0],[0],[1],[1]])

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
first_layer_weights = 2 * np.random.random((3,1)) - 1

for iter in range(1000):
  # forward propagation
  input_layer = x # give input
  output_layer = nonlin(
      np.dot(input_layer, first_layer_weights)
    )
  
  # how much did we miss?
  error = y - output_layer

  # multiply how much we missed by the
  # slope of the sigmoid at the values in l1
  out_delta = error * nonlin(output_layer, True)

  # update weights
  first_layer_weights += np.dot(input_layer.T, out_delta)

print("Output After Training:")
print(output_layer)