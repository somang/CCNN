from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation
from keras.utils import plot_model, np_utils
from keras.models import load_model
from keras.optimizers import SGD

from sklearn.linear_model import Perceptron
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
mpl.rcParams['agg.path.chunksize'] = 10000

SCALE = 5 # for categorical filtering
DATAFILE = str(SCALE) + "_gen_dt_100000.csv"
#DATAFILE = str(SCALE) + "_nd_dt_100000.csv"
MODEL_FILE = "emp_model.h5" 

def data_prep():
  ######################## data prep ###########################################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  t_index = round(len(dataset)*0.8) # 80% to train
  # normalization and prep the data for train and validation
  # values: delay, wpm, sge, mw, ss, pf
  # score:  delay, speed, sge, mw, verbatim
  np.set_printoptions(precision=4, suppress=True)
  sc = StandardScaler()
  x = sc.fit_transform(dataset[:, :6])
  #x = dataset[:,:6]
  x_emp_tr = x[:t_index, :4]
  x_emp_ts = x[t_index:, :4]
  
  x_ver_tr = x[:t_index, 3:6]
  x_ver_ts = x[t_index:, 3:6]
  
  y_emp_lm_tr = dataset[:t_index, 6:10] # regression for training
  y_emp_lm_ts = dataset[t_index:, 6:10] # regression for test
  y_ver_lm_tr = dataset[:t_index, 10:11] # regression for training  
  y_ver_lm_ts = dataset[t_index:, 10:11] # regression for test

  if SCALE == 10:
    flag = 55
  elif SCALE == 5:
    flag = 31
  elif SCALE == 3:
    flag = 23  
  y_emp_nn_tr = dataset[:t_index, 11:flag] # categorical for training
  y_emp_nn_ts = dataset[t_index:, 11:flag] # categorical for test
  y_ver_nn_tr = dataset[:t_index, flag:] # categorical for training
  y_ver_nn_ts = dataset[t_index:, flag:] # categorical for test

  '''
  # draw data
  plt.plot(ver_tr_x[:,0], ver_tr_y[:,0], c='r')
  plt.xlabel('sentence similarity')
  plt.ylabel('verbatimness')
  plt.show()

  plt.scatter(ver_tr_x[:,1], ver_tr_y[:,0], c='g')
  plt.xlabel('number of missing words')
  plt.ylabel('verbatimness')
  plt.show()

  plt.scatter(ver_tr_x[:,2], ver_tr_y[:,0], c='b')
  plt.xlabel('pf factors')
  plt.ylabel('verbatimness')
  plt.show()
  '''
  
  return (x_emp_tr, x_emp_ts, x_ver_tr, x_ver_ts, 
         y_emp_lm_tr, y_emp_lm_ts, y_ver_lm_tr, y_ver_lm_ts, 
         y_emp_nn_tr, y_emp_nn_ts, y_ver_nn_tr, y_ver_nn_ts)

def draw_graphs(hist):
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
  

def print_model_perf(tst_x, tst_y, predictions, name):
  rms = sqrt(mean_squared_error(predictions, tst_y))
  print(name + " RMSE : {:.2f}".format(rms))
  print()
  # plot the graph prediction vs real value
  scatter2 = plt.scatter(tst_y, predictions)
  plt.ylabel('predictions')
  plt.xlabel('Real values')
  plt.show()

def baseline_model(output_unit, loss, output_activation):
  # create model
  model = Sequential()
  model.add(Dense(4, input_dim=3, kernel_initializer='glorot_uniform', activation='relu'))
  model.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))
  model.add(Dense(output_unit, kernel_initializer='glorot_uniform', activation=output_activation))
  # Compile model
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
  model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
  #model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy'])
  #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  return model

if __name__ == '__main__':
  # data prep
  (x_emp_tr, x_emp_ts, x_ver_tr, x_ver_ts, 
  y_emp_lm_tr, y_emp_lm_ts, y_ver_lm_tr, y_ver_lm_ts, 
  y_emp_nn_tr, y_emp_nn_ts, y_ver_nn_tr, y_ver_nn_ts) = data_prep()
  # create model
  regression_model = baseline_model(1, 'mse', 'relu')

  print("TRAINING: Regression model")
  reg_hist = regression_model.fit(x_ver_tr, y_ver_lm_tr, 
              epochs=30, batch_size=500,
              verbose=1, validation_data=(x_ver_ts, y_ver_lm_ts)
              )
  # evaluate the regression value model
  #draw_graphs(reg_hist)
  predictions = regression_model.predict(x_ver_ts, batch_size=10)
  rms = sqrt(mean_squared_error(y_ver_lm_ts, predictions))
  print("NN RMSE: {:.2f}".format(rms))
  print_model_perf(x_ver_ts, y_ver_lm_ts, predictions, "NN")

  #save the model
  regression_model.save('reg_ver_model.h5') #creates a hdf5 file


  ############################## multivariate linear regression
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(ver_tr_x, ver_tr_y[:,-1:])
  predictions = mlm.predict(ver_tst_x)
  rms = sqrt(mean_squared_error(ver_tst_y, predictions))
  print("MLM RMSE: {:.2f}".format(rms))
  print_model_perf(ver_tst_x, ver_tst_y, predictions, "MLM")

  ############################## Polynomial linear regression
  poly = PolynomialFeatures(degree=4)
  training_x = poly.fit_transform(ver_tr_x)
  testing_x = poly.fit_transform(ver_tst_x)
  lg = linear_model.LinearRegression()
  lg.fit(training_x, ver_tr_y)
  predictions = lg.predict(testing_x)
  rms = sqrt(mean_squared_error(ver_tst_y, predictions))
  print("MPM RMSE: {:.2f}".format(rms))
  print_model_perf(ver_tst_x, ver_tst_y, predictions, "MPM")

  ############################## Support Vector Machine?