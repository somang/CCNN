from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout
from keras.utils import plot_model

from scipy import stats
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

import numpy as np
from math import sqrt


SCALE = 5 # for categorical filtering
DATAFILE = "data/" + str(SCALE) + "_gen_dt_100000.csv"
#DATAFILE = "data/" + str(SCALE) + "_nd_dt_100000.csv"
#DATAFILE = "old_data/" + str(SCALE) + "_nd_dt_100000.csv"

TRAINING = 1
MODEL_FILE = "emp_model.h5" 
EPOCHS = 100

def data_prep():
  ######################## data prep ###########################################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  t_index = round(len(dataset)*0.8) # 80% to train
  # normalization and prep the data for train and validation
  # 6 values: delay, wpm, sge, mw, ss, pf
  # 5 scores to be predicted:  delay, speed, sge, mw, verbatim
  np.set_printoptions(precision=4, suppress=True)
  sc = StandardScaler()
  x = sc.fit_transform(dataset[:, :6])
  x_tr = x[:t_index, :6]
  x_ts = x[t_index:, :6]
  y_tr = dataset[:t_index, 6:11] # regression for training
  y_ts = dataset[t_index:, 6:11] # regression for test
  return (x_tr, x_ts, y_tr, y_ts)
  
def train_nn_regression(x_tr, y_tr, x_ts, y_ts):
  # create model using sequential
  model = Sequential()
  model.add(Dense(units=64, input_dim=6, activation='relu'))
  model.add(Dense(units=32, activation='relu'))
  #model.add(Dense(units=32, activation='relu'))
  model.add(Dense(units=5))
  # compile
  model.compile(loss='mean_squared_error',
                optimizer='adam', 
                metrics=['accuracy'])
  # fit the model
  hist = model.fit(x_tr, y_tr, 
                epochs=EPOCHS, batch_size=500,
                verbose=2, validation_data=(x_ts, y_ts)
                )
  #save the model
  model.save(MODEL_FILE) #creates a hdf5 file
  return model, hist

def print_model_perf(predictions, x_ts, y_ts, name):
  category_set = ["Delay", "Speed", "Spelling and Grammar", "Missing Words", "Verbatim accuracy"]
  col_set = ['g','b','y','c','r']
  for i in range(5):
    color = col_set[i]
    category = category_set[i]
    rms = sqrt(mean_squared_error(y_ts[:,i], predictions[:,i]))
    print(name + " " + category_set[i] + "\nRMSE: {:.2f}".format(rms))
    correct = 0
    rounded_p = np.rint(predictions)
    for j in range(len(rounded_p)):
      if rounded_p[j,i] == y_ts[j,i]:
        correct += 1
    #print(correct,"correct answers out of",len(rounded_p))
    print("acc: {:.2f}%".format(correct/len(rounded_p[:,i])*100.0))
  print()

if __name__ == '__main__':
  print(DATAFILE)
  if TRAINING:
    (x_tr, x_ts, y_tr, y_ts) = data_prep()
    model, hist = train_nn_regression(x_tr, y_tr, x_ts, y_ts)

  # Graph the comparison between prediction vs real
  predictions = model.predict(x_ts, batch_size=10)
  print_model_perf(predictions, x_ts, y_ts, "Multilayer Perceptron:")

  ##### Multivariate linear regression
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(x_tr, y_tr)
  predictions = mlm.predict(x_ts)
  print_model_perf(predictions, x_ts, y_ts, "MLM")
  
  ##### Polynomial linear regression
  poly = PolynomialFeatures(degree=3)
  training_x = poly.fit_transform(x_tr)
  testing_x = poly.fit_transform(x_ts)
  lg = linear_model.LinearRegression()
  lg.fit(training_x, y_tr)
  predictions = lg.predict(testing_x)
  print_model_perf(predictions, x_ts, y_ts, "Polynomial Regression:")




"""
Multilayer Perceptron: Delay
RMSE: 0.28
acc: 88.84%
Multilayer Perceptron: Speed
RMSE: 0.51
acc: 58.75%
Multilayer Perceptron: Spelling and Grammar
RMSE: 0.63
acc: 61.69%
Multilayer Perceptron: Missing Words
RMSE: 0.64
acc: 64.22%
Multilayer Perceptron: Verbatim accuracy
RMSE: 0.25
acc: 91.53%


"""