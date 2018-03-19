from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation
from keras.utils import plot_model, np_utils
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
mpl.rcParams['agg.path.chunksize'] = 10000

DATAFILE = "ver_gen_dt_100000.csv"
MODEL_FILE = "ver_model.h5" 

def data_prep():
  ######################## data prep ###########################################
  dataset = np.loadtxt(DATAFILE, delimiter=",")
  dataset = dataset[:10000,:]
  t_index = round(len(dataset)*0.8)

  X_data_ver, Y_data_ver = dataset[:,:3], dataset[:,13:]
  #sc = StandardScaler()
  #X_data_ver = sc.fit_transform(X_data_ver)

  # split input(X) and output(Y)
  ver_tr_x = X_data_ver[:t_index,:]
  ver_tr_y = Y_data_ver[:t_index,:]
  
  ver_tst_x = X_data_ver[t_index:,:]
  ver_tst_y = Y_data_ver[t_index:,:]

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

  return ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y

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
  
def baseline_model(optimizer='rmsprop', init='glorot_uniform'):
  # create model
  model = Sequential()
  model.add(Dense(64, input_dim=3, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1, activation='relu'))
  # Compile model
  #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
  #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  return model

if __name__ == '__main__':
  # data prep
  ver_tr_x, ver_tr_y, ver_tst_x, ver_tst_y = data_prep()



  # create model
  #model = KerasClassifier(build_fn=baseline_model, verbose=1)
  model = baseline_model()
  '''
  ver_hist = model.fit(ver_tr_x, ver_tr_y, 
              epochs=150, batch_size=500,
              verbose=1, validation_data=(ver_tst_x, ver_tst_y)
              )
  
  #save the model
  model.save('ver_model.h5') #creates a hdf5 file
  # evaluate the empirical value model
  loss_and_metrics = model.evaluate(ver_tst_x, ver_tst_y, batch_size=50)
  print("\n%s: %.2f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))
  draw_graphs(ver_hist)
  
  predictions = model.predict(ver_tst_x, batch_size=10)
  plt.plot(ver_tst_y[:,0], predictions[:,0], c='r')
  #plt.scatter(ver_tst_y[:,0], predictions[:,0])
  plt.xlabel('real values')
  plt.ylabel('predictions')
  plt.title('Neural Networks Score predictions')
  plt.legend(['real values', 'predictions'])
  plt.show()
  

  ############################## multivariate linear regression
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(ver_tr_x, ver_tr_y)
  predictions = mlm.predict(ver_tst_x)
  ## plot the mlm
  plt.plot(ver_tst_y, predictions, c='c')
  #plt.scatter(ver_tst_y, predictions, c='r')
  plt.title('Linear Regression Score predictions')
  plt.xlabel("Real Values")
  plt.ylabel("Predictions")
  plt.show()
  # print the accuracy score
  print("Score:", mlm.score(ver_tst_x, ver_tst_y))
  
  ############################## Polynomial linear regression
  poly = PolynomialFeatures(degree=2)
  training_x = poly.fit_transform(ver_tr_x)
  testing_x = poly.fit_transform(ver_tst_x)
  lg = linear_model.LinearRegression()
  lg.fit(training_x, ver_tr_y)
  predictions = lg.predict(testing_x)
  plt.plot(ver_tst_y, predictions, c='b')
  plt.title('Polynomial Regression Score predictions')
  plt.xlabel("Real Values")
  plt.ylabel("Predictions")
  plt.show()
  '''

  ############################## Logistic Regression