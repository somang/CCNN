from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.utils import plot_model
from keras.models import load_model

from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
mpl.rcParams['agg.path.chunksize'] = 10000

DATAFILE = "5_gen_dt_100000.csv"
MODEL_FILE = "emp_model.h5" 
TRAINING = 1
EPOCHS = 150
VS_SWITCH = 1

def data_prep():
  ######################## data prep ###########################################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  t_index = round(len(dataset)*0.8) # 80% to train
  # normalization and prep the data for train and validation
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
  ######################## data prep ###########################################
  return emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y, ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y


def train_nn(emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y):
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
                verbose=0, validation_data=(emp_tst_x, emp_tst_y)
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

if __name__ == '__main__':
  if TRAINING:
    emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y, ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y = data_prep()
    emp_model,hist = train_nn(emp_tr_x, emp_tr_y, emp_tst_x, emp_tst_y)
    #draw_graphs(hist)

  # graph the comparison between prediction vs real
  predictions = emp_model.predict(emp_tst_x, batch_size=10)
  predictions = np.rint(predictions)

  category_set = ["Delay", "Speed", "Spelling and Grammar", "Missing Words"]
  col_set = ['g','b','y','c']
  for i in range(4):
    color = col_set[i]
    category = category_set[i]
    cat = predictions[:,i]
    tst = emp_tst_y[:,i]
    correct_count = 0
    for j in range(len(cat)):
      p_value, t_value = cat[j], tst[j]
      if p_value == t_value:
        correct_count += 1
    print("NN accuracy on " + category_set[i] + ": {:.2f}%".format(correct_count/len(cat)*100))

    '''
    if VS_SWITCH:
      #plt.plot(emp_tst_y[:,i], predictions[:,i], c=color)
      plt.scatter(emp_tst_y[:,i], predictions[:,i], c=color)
      plt.xlabel('real values')
      plt.ylabel('predictions')
      plt.title(category + ' Score predictions')
      plt.legend(['real values', 'predictions'])
      plt.show()
    else:
      plt.plot(emp_tst_x[:,i], emp_tst_y[:,i], c=color) # real values
      plt.plot(emp_tst_x[:,i], predictions[:,i], c='r') # predictions
      plt.xlabel('Input X')
      plt.ylabel('Scores')
      plt.title(category + ' Score comparison')
      plt.legend(['Input X', 'Scores'])
      plt.show()
    '''

  ##### multivariate nonlinear regression fitting
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(emp_tr_x, emp_tr_y)
  predictions = mlm.predict(emp_tst_x)
  predictions = np.rint(predictions)
  
  # print the accuracy score
  category_set = ["Delay", "Speed", "Spelling and Grammar", "Missing Words"]
  col_set = ['g','b','y','c']
  for i in range(4):
    color = col_set[i]
    category = category_set[i]
    cat = predictions[:,i]
    tst = emp_tst_y[:,i]
    correct_count = 0
    for j in range(len(cat)):
      p_value, t_value = cat[j], tst[j]
      if p_value == t_value:
        correct_count += 1
    print("MLM accuracy on " + category_set[i] + ": {:.2f}%".format(correct_count/len(cat)*100))

    ## plot the mlm
    '''
    for i in range(4):
      #plt.plot(emp_tst_y[:,i], predictions[:,i], c='c')
      plt.scatter(emp_tst_y[:,i], predictions[:,i])
      plt.title('Linear Regression Score predictions')
      plt.xlabel("Real Values")
      plt.ylabel("Predictions")
      plt.show()
      # print the accuracy score
    '''

