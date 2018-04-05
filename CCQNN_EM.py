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
from scipy.stats.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats.mstats import spearmanr

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
mpl.rcParams['agg.path.chunksize'] = 10000

SCALE = 10 # for categorical filtering
DATAFILE = str(SCALE) + "_gen_dt_100000.csv"
#DATAFILE = str(SCALE) + "_nd_dt_100000.csv"
MODEL_FILE = "emp_model.h5" 
TRAINING = 1
EPOCHS = 80

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

  return (x_emp_tr, x_emp_ts, x_ver_tr, x_ver_ts, 
         y_emp_lm_tr, y_emp_lm_ts, y_ver_lm_tr, y_ver_lm_ts, 
         y_emp_nn_tr, y_emp_nn_ts, y_ver_nn_tr, y_ver_nn_ts)
  
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
                verbose=2, validation_data=(emp_tst_x, emp_tst_y)
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
  emp_model.add(Dense(units=64, input_dim=4, activation='sigmoid'))
  emp_model.add(Dense(units=32, activation='sigmoid'))
  emp_model.add(Dense(units=output_size, activation='sigmoid'))
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
  #loss_and_metrics = emp_model.evaluate(emp_tst_x, emp_tst_y, batch_size=50)
  #print("\n%s: %.2f%%" % (emp_model.metrics_names[1], loss_and_metrics[1]*100))
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
    print(name + " " + category_set[i] + "\nRMSE: {:.2f}".format(rms))
    correct = 0
    rounded_p = np.rint(predictions)
    for j in range(len(rounded_p)):
      if rounded_p[j,i] == tst_y[j,i]:
        correct += 1
    #print(correct,"correct answers out of",len(rounded_p))
    print("acc: {:.2f}%".format(correct/len(rounded_p[:,i])*100.0))
    
    # plot the graph prediction vs real value
    #plot_pred(tst_x[:,i], tst_y[:,i], predictions[:,i], color, category)
  print()

if __name__ == '__main__':
  print(DATAFILE)
  if TRAINING:
    (x_emp_tr, x_emp_ts, x_ver_tr, x_ver_ts, 
        y_emp_lm_tr, y_emp_lm_ts, y_ver_lm_tr, 
        y_ver_lm_ts, y_emp_nn_tr, y_emp_nn_ts, 
        y_ver_nn_tr, y_ver_nn_ts) = data_prep()
    #emp_model, hist = train_nn_categorical(x_emp_tr, y_emp_nn_tr, x_emp_ts, y_emp_nn_ts)
    emp_model, hist = train_nn_regression(x_emp_tr, y_emp_lm_tr, x_emp_ts, y_emp_lm_ts)

    #draw_graphs(hist)

  # Graph the comparison between prediction vs real
  predictions = emp_model.predict(x_emp_ts, batch_size=10)
  print_model_perf(predictions, x_emp_ts, y_emp_lm_ts, "Multilayer Perceptron:")

  ###### testing correlation among the variables
  delay_x, speed_x, sge_x, mw_x, ss_x, pf_x = x_emp_tr[:,0], x_emp_tr[:,1], x_emp_tr[:,2], x_emp_tr[:,3], x_ver_tr[:,1], x_ver_tr[:,2]
  delay_score, speed_score, sge_score, mw_score, verbatim_score = y_emp_lm_tr[:,0], y_emp_lm_tr[:,1], y_emp_lm_tr[:,2], y_emp_lm_tr[:,3], y_ver_lm_tr[:,0]
  '''
  # pearson's r and 2 tailed p value
  print(pearsonr(mw_x, ss_x)) # -0.243, 0.0             [V]
  print(pearsonr(delay_x, delay_score)) # -0.812, 0     [V]
  print(pearsonr(speed_x, speed_score)) # -0.647, 0     [V]
  print(pearsonr(sge_x, sge_score)) # -0.823, 0         [V]
  print(pearsonr(mw_x, mw_score)) # -0.544, 0           [V]
  print(pearsonr(ss_x, verbatim_score)) # 0.668, 0      [V]
  print(pearsonr(mw_x, verbatim_score)) # -0.564, 0     [V]
  # Spearman's rho and 2 tailed p value
  print(spearmanr(delay_x, delay_score)) # -0.557, 0
  print(spearmanr(speed_x, speed_score)) # -0.393, 0
  print(spearmanr(sge_x, sge_score)) # -0.699, 0
  print(spearmanr(mw_x, mw_score)) # -0.699, 0
  print(spearmanr(ss_x, verbatim_score)) # 0.401, 0
  print(spearmanr(mw_x, verbatim_score)) # -0.152, 0
  '''
  # Kendall tau
  print(kendalltau(delay_x, delay_score)) # -0.420, 0
  print(kendalltau(speed_x, speed_score)) # -0.294, 0     [V]
  print(kendalltau(sge_x, sge_score)) # -0.599, 0         [V]
  print(kendalltau(mw_x, mw_score)) # -0.546, 0           [V]
  print(kendalltau(ss_x, verbatim_score)) # 0.302, 0      [V]
  print(kendalltau(mw_x, verbatim_score)) # -0.119, 0     [V]
  
  
  #####
  # It can be concluded that there exists no linear correlation between factors
  # However, it showed that each of the factor-score relation is linearly correlated.
  # Therefore, there should be a linear regression model for each factor-score relationship.
  ##### 

  '''
  # Simple linear regression model for delay factor-score
  mlm = linear_model.LinearRegression()
  delay_x = [delay_x]
  delay_x = np.array(delay_x).reshape(-1, 1)
  delay_score = [delay_score]
  delay_score = np.array(delay_score).reshape(-1, 1)
  delay_x_tst = [x_emp_ts[:,0]]
  delay_x_tst = np.array(delay_x_tst).reshape(-1, 1)

  stat_model = mlm.fit(delay_x, delay_score)
  predictions = mlm.predict(delay_x_tst)
  rms = sqrt(mean_squared_error(y_emp_lm_ts[:,0], predictions))
  print("Delay linear model RMSE: {:.2f}".format(rms))
  correct = 0
  rounded_p = np.rint(predictions)
  for j in range(len(rounded_p)):
    if rounded_p[j,0] == y_emp_lm_ts[j,0]:
      correct += 1
  print("Delay linear model acc: {:.2f}%".format(correct/len(rounded_p[:,0])*100.0))

  # Simple linear regression model for speed factor-score
  mlm = linear_model.LinearRegression()
  speed_x = [speed_x]
  speed_x = np.array(speed_x).reshape(-1, 1)
  speed_score = [speed_score]
  speed_score = np.array(speed_score).reshape(-1, 1)
  speed_x_tst = [x_emp_ts[:,1]]
  speed_x_tst = np.array(speed_x_tst).reshape(-1, 1)

  stat_model = mlm.fit(speed_x, speed_score)
  predictions = mlm.predict(speed_x_tst)
  rms = sqrt(mean_squared_error(y_emp_lm_ts[:,1], predictions))
  print("Speed linear model RMSE: {:.2f}".format(rms))
  correct = 0
  rounded_p = np.rint(predictions)
  for j in range(len(rounded_p)):
    if rounded_p[j,0] == y_emp_lm_ts[j,1]:
      correct += 1
  print("Speed linear model acc: {:.2f}%".format(correct/len(rounded_p[:,0])*100.0))
  '''

  ##### Multivariate linear regression
  mlm = linear_model.LinearRegression()
  stat_model = mlm.fit(x_emp_tr, y_emp_lm_tr)
  predictions = mlm.predict(x_emp_ts)
  #print_model_perf(predictions, x_emp_ts, y_emp_lm_ts, "MLM")
  
  ##### Polynomial linear regression
  poly = PolynomialFeatures(degree=3)
  training_x = poly.fit_transform(x_emp_tr)
  testing_x = poly.fit_transform(x_emp_ts)
  lg = linear_model.LinearRegression()
  lg.fit(training_x, y_emp_lm_tr)
  predictions = lg.predict(testing_x)
  print_model_perf(predictions, x_emp_ts, y_emp_lm_ts, "Polynomial Regression:")
