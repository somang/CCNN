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

#DATAFILE = "10_gen_dt_100000.csv"
DATAFILE = "10_nd_dt_100000.csv"
MODEL_FILE = "ver_model.h5" 

def data_prep():
  ######################## data prep #################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  dataset = dataset[:100000,:]
  t_index = round(len(dataset)*0.8)
  # normalize data, prep for train and validation
  # values: delay, wpm, sge, mw, ss, pf
  # score:  delay, speed, sge, mw, verbatim
  sc = StandardScaler()
  X_data = sc.fit_transform(dataset[:,:6])
  X_data_emp, Y_data_emp = X_data[:,:4], dataset[:,6:10]
  X_data_ver, Y_data_ver = X_data[:,3:6], dataset[:,10:]
  # split input(X) and output(Y)
  emp_tr_x = X_data_emp[:t_index,:]
  emp_tr_y = Y_data_emp[:t_index,:]
  emp_tst_x = X_data_emp[t_index:,:]
  emp_tst_y = Y_data_emp[t_index:,:]

  ver_tr_x = X_data_ver[:t_index,:]
  ver_tr_y = Y_data_ver[:t_index,:]
  ver_tst_x = X_data_ver[t_index:,:]
  ver_tst_y = Y_data_ver[t_index:,:]
  ######################## data prep ################
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
  return emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y, ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y

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
  
def baseline_model(output_unit, loss, output_activation):
  # create model
  model = Sequential()
  model.add(Dense(64, input_dim=3, kernel_initializer='glorot_uniform', activation='relu'))
  model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
  model.add(Dense(output_unit, kernel_initializer='glorot_uniform', activation=output_activation))
  # Compile model
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
  model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
  #model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy'])
  #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  return model

if __name__ == '__main__':
  # data prep
  emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y, ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y = data_prep()
  # create model
  regression_model = baseline_model(1, 'mse', 'relu')

  print("TRAINING: Regression model")
  reg_hist = regression_model.fit(ver_tr_x, ver_tr_y, 
              epochs=50, batch_size=500,
              verbose=0, validation_data=(ver_tst_x, ver_tst_y)
              )
  # evaluate the regression value model
  #draw_graphs(reg_hist)
  predictions = regression_model.predict(ver_tst_x, batch_size=10)
  rms = sqrt(mean_squared_error(ver_tst_y, predictions))
  print("NN MSE: {:.2f}%".format(rms))
  
  #save the model
  regression_model.save('reg_ver_model.h5') #creates a hdf5 file

  '''
  #plt.plot(ver_tst_y, predictions, c='r')
  plt.scatter(ver_tst_y, predictions)
  plt.xlabel('real values')
  plt.ylabel('predictions')
  plt.title('Neural Networks Score predictions')
  plt.show()
  '''

  ############################## multivariate linear regression
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(ver_tr_x, ver_tr_y[:,-1:])
  predictions = mlm.predict(ver_tst_x)
  rms = sqrt(mean_squared_error(ver_tst_y, predictions))
  print("MLM MSE: {:.2f}%".format(rms))

  '''
  # plot the mlm
  #plt.plot(ver_tst_y, predictions, c='c')
  plt.scatter(ver_tst_y, predictions, c='r')
  plt.title('Linear Regression Score predictions')
  plt.xlabel("Real Values")
  plt.ylabel("Predictions")
  plt.show()
  '''

  ############################## Polynomial linear regression
  poly = PolynomialFeatures(degree=2)
  training_x = poly.fit_transform(ver_tr_x)
  testing_x = poly.fit_transform(ver_tst_x)
  lg = linear_model.LinearRegression()
  lg.fit(training_x, ver_tr_y)
  predictions = lg.predict(testing_x)
  rms = sqrt(mean_squared_error(ver_tst_y, predictions))
  print("MPM MSE: {:.2f}%".format(rms))

  #predictions = np.rint(predictions)
  # print the accuracy score
  #correct_count = 0
  #for i in range(len(predictions)):
  #  if predictions[i][0] == ver_tst_y[i][0]:
  #    correct_count += 1
  #print("Polynomial regression accuracy: {:.2f}%".format(correct_count/len(predictions)*100))

  '''
  ## plot the lg
  #plt.plot(ver_tst_y[:,-1:], predictions, c='b')
  plt.scatter(ver_tst_y, predictions, c='b')
  plt.title('Polynomial Regression Score predictions')
  plt.xlabel("Real Values")
  plt.ylabel("Predictions")
  plt.show()
  '''
  
  ############################## Support Vector Machine?