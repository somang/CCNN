import csv
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

gen_data = []
#https://crtc.gc.ca/eng/archive/2012/2012-362.htm
#status, delay, wpm, similarity, number of errors, 
# speed, delay, missing words, grammer errors, verbatim

rand_delay = np.random.uniform(low=0.0, high=10000.0, size=(3000,1))
rand_speed = np.random.uniform(low=0.0, high=500.0, size=(3000,1))
rand_misspelled = np.random.randint(10, size=(3000,1))
rand_sentence

#c = np.column_stack((rand_delay, rand_speed))

'''
with open('generated_data.csv', 'w') as mf:
  wr = csv.writer(mf, quoting=csv.QUOTE_ALL)
  wr.writerow(gen_data)
'''