from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.utils import plot_model
from keras.models import load_model

from scipy import stats
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
mpl.rcParams['agg.path.chunksize'] = 10000

#DATAFILE = "10_gen_dt_100000.csv"
SCALE = 10
DATAFILE = "10_nd_dt_100000.csv"
MODEL_FILE = "emp_model.h5" 
TRAINING = 1
EPOCHS = 50

def data_prep():
  ######################## data prep ###########################################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  t_index = round(len(dataset)*0.01) # 80% to train
  # normalization and prep the data for train and validation
  # values: delay, wpm, sge, mw, ss, pf
  # score:  delay, speed, sge, mw, verbatim
  np.set_printoptions(precision=4, suppress=True)
  sc = StandardScaler()
  x = sc.fit_transform(dataset[:, :6])
  
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

  return x_emp_tr, x_emp_ts, x_ver_tr, x_ver_ts, 
         y_emp_lm_tr, y_emp_lm_ts, y_ver_lm_tr, y_ver_lm_ts, 
         y_emp_nn_tr, y_emp_nn_ts, y_ver_nn_tr, y_ver_nn_ts
  
def train_nn_regression(emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y):
  # create model using sequential
  emp_model = Sequential()
  emp_model.add(Dense(units=64, input_dim=4, activation='relu'))
  emp_model.add(Dense(units=32, activation='relu'))
  emp_model.add(Dense(units=4))
  # compile
  emp_model.compile(loss='mean_squared_error',
                optimizer='adam', 
                metrics=['accuracy'])
  # fit the empirical value model
  emp_hist = emp_model.fit(emp_tr_x, emp_tr_y, 
                epochs=EPOCHS, batch_size=500,
                verbose=1, validation_data=(emp_tst_x, emp_tst_y)
                )
  #save the model
  emp_model.save(MODEL_FILE) #creates a hdf5 file
  # evaluate the empirical value model
  #loss_and_metrics = emp_model.evaluate(emp_tst_x, emp_tst_y, batch_size=50)
  #print("\n%s: %.2f%%" % (emp_model.metrics_names[1], loss_and_metrics[1]*100))
  return emp_model, emp_hist

def train_nn_categorical(emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y):
  output_size = 44 if SCALE == 10 else SCALE*4
  # create model using sequential
  emp_model = Sequential()
  emp_model.add(Dense(units=64, input_dim=4, activation='relu'))
  emp_model.add(Dense(units=32, activation='relu'))
  emp_model.add(Dense(units=output_size))
  # compile
  emp_model.compile(loss='categorical_crossentropy',
                optimizer='adam', 
                metrics=['accuracy'])
  # fit the empirical value model
  emp_hist = emp_model.fit(emp_tr_x, emp_tr_y, 
                epochs=EPOCHS, batch_size=500,
                verbose=1, validation_data=(emp_tst_x, emp_tst_y)
                )
  #save the model
  emp_model.save(MODEL_FILE) #creates a hdf5 file
  # evaluate the empirical value model
  loss_and_metrics = emp_model.evaluate(emp_tst_x, emp_tst_y, batch_size=50)
  print("\n%s: %.2f%%" % (emp_model.metrics_names[1], loss_and_metrics[1]*100))
  return emp_model, emp_hist

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

def plot_pred(x, y, pred, color, category):
  scatter1 = plt.scatter(x, pred, c='r', label="predictions")
  scatter2 = plt.scatter(x, y, c=color, label="real values")
  plt.ylabel('output Y values')
  plt.xlabel('input X values')
  plt.title(category + ' Score predictions')
  plt.legend(handles=[scatter1, scatter2], loc = 0)
  plt.show()

def print_model_perf(predictions, tst_x, tst_y, name):
  category_set = ["Delay", "Speed", "Spelling and Grammar", "Missing Words"]
  col_set = ['g','b','y','c']
  for i in range(4):
    color = col_set[i]
    category = category_set[i]
    rms = sqrt(mean_squared_error(tst_y[:,i], predictions[:,i]))
    print(name + category_set[i] + ": {:.2f}".format(rms))
    # plot the graph prediction vs real value
    #plot_pred(tst_x[:,i], tst_y[:,i], predictions[:,i], color, category)
  print()

if __name__ == '__main__':
  if TRAINING:
    x_emp_tr, x_emp_ts, x_ver_tr, x_ver_ts, y_emp_lm_tr, y_emp_lm_ts, y_ver_lm_tr, y_ver_lm_ts, y_emp_nn_tr, y_emp_nn_ts, y_ver_nn_tr, y_ver_nn_ts = data_prep()
    emp_model, hist = train_nn(x_emp_tr, y_emp_nn_tr, x_emp_ts, y_emp_nn_ts)
    draw_graphs(hist)

  



  '''
  # Graph the comparison between prediction vs real
  predictions = emp_model.predict(emp_tst_x, batch_size=10)
  rms = sqrt(mean_squared_error(emp_tst_y, predictions))
  print("NN RMSE: {:.2f}".format(rms))
  print_model_perf(predictions, emp_tst_x, emp_tst_y, "NN")
  
  ##### Multivariate linear regression
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(emp_tr_x, emp_tr_y)
  predictions = mlm.predict(emp_tst_x)
  rms = sqrt(mean_squared_error(emp_tst_y, predictions))
  print("MLM RMSE: {:.2f}".format(rms))
  print_model_perf(predictions, emp_tst_x, emp_tst_y, "MLM")

  ##### Polynomial linear regression
  poly = PolynomialFeatures(degree=4)
  training_x = poly.fit_transform(emp_tr_x)
  testing_x = poly.fit_transform(emp_tst_x)
  lg = linear_model.LinearRegression()
  lg.fit(training_x, emp_tr_y)
  predictions = lg.predict(testing_x)
  rms = sqrt(mean_squared_error(emp_tst_y, predictions))
  print("MPM RMSE: {:.2f}".format(rms))
  print_model_perf(predictions, emp_tst_x, emp_tst_y, "MPM")
  '''
