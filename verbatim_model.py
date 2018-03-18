from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation
from keras.utils import plot_model
from keras.models import load_model

from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
mpl.rcParams['agg.path.chunksize'] = 10000

DATAFILE = "gen_dt_100000.csv"
MODEL_FILE = "ver_model.h5" 

def data_prep():
  ######################## data prep ###########################################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  t_index = round(len(dataset)*0.8)
  # normalization and prep the data for train and validation
  sc = StandardScaler()
  X_data = sc.fit_transform(dataset[:,:6])
  X_data_emp, Y_data_emp = X_data[:,:4], dataset[:,6:10]
  X_data_ver, Y_data_ver = X_data[:,3:6], dataset[:,10:11]

  # split input(X) and output(Y)
  emp_tr_x = X_data_emp[:t_index,:]
  emp_tr_y = Y_data_emp[:t_index,:]
  emp_tst_x = X_data_emp[t_index:,:]
  emp_tst_y = Y_data_emp[t_index:,:]

  ver_tr_x = X_data_ver[:t_index,:]
  ver_tr_y = Y_data_ver[:t_index,:]
  ver_tst_x = X_data_ver[t_index:,:]
  ver_tst_y = Y_data_ver[t_index:,:]
  ######################## data prep ###########################################
  return emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y,ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y

def train_nn(ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y):
  # create model using sequential
  ver_model = Sequential()
  ver_model.add(Dense(units=128, input_dim=3, activation='relu'))
  #ver_model.add(Dropout(.2))
  ver_model.add(Dense(units=64, activation='relu'))
  #ver_model.add(Dropout(.2))
  ver_model.add(Dense(units=1))

  # compile
  ver_model.compile(loss='mean_squared_error',
                optimizer='adam', 
                metrics=['accuracy'])
  # fit the empirical value model
  ver_hist = ver_model.fit(ver_tr_x, ver_tr_y, 
                epochs=150, batch_size=250,
                verbose=1, validation_data=(ver_tst_x, ver_tst_y)
                )
  #save the model
  ver_model.save(MODEL_FILE) #creates a hdf5 file
  # evaluate the empirical value model
  loss_and_metrics = ver_model.evaluate(ver_tst_x, ver_tst_y, batch_size=50)
  print("\n%s: %.2f%%" % (ver_model.metrics_names[1], loss_and_metrics[1]*100))
  return ver_model, ver_hist

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

if __name__ == '__main__':
  emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y,ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y = data_prep()
  ver_model,hist = train_nn(ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y)
  draw_graphs(hist)

  # graph the comparison between prediction vs real
  predictions = ver_model.predict(ver_tst_x, batch_size=10)
  plt.plot(ver_tst_y[:,0], np.rint(predictions[:,0]), c='c')

  plt.xlabel('real values')
  plt.ylabel('predictions')
  plt.title('Verbatim Score predictions')
  plt.legend(['real values', 'predictions'])
  plt.show()
