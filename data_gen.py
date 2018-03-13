import csv
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from random import randint


gen_data = []
#https://crtc.gc.ca/eng/archive/2012/2012-362.htm


# delay, wpm, similarity, number of errors
rand_delay = np.random.uniform(low=0.0, high=10000.0, size=(3000,1))
rand_wpm = np.random.uniform(low=0.0, high=500.0, size=(3000,1))
rand_sentence_sim = np.random.uniform(low=0.0, high=100.0, size=(3000,1))
rand_errors = np.random.randint(10, size=(3000,1))


c = np.column_stack((rand_delay, rand_wpm))
c = np.column_stack((c, rand_sentence_sim))
c = np.column_stack((c, rand_errors))

#print(c)
# speed, delay, missing words, grammar errors, verbatim
rating_list = [[],[],[],[],[]]
for i in c:
  print(i)
  speed_rating, delay_rating, misspell_rating, grammar_rating, verbatim_rating = 0,0,0,0,0

  # calculate delay rating
  delay = i[0]
  if delay <= 100:
    delay_rating = 10
  elif 100 < delay <= 500:
    delay_rating = randint(8,9)
  elif 500 < delay <= 2000:
    delay_rating = randint(6, 9)
  elif 2000 < delay <= 4000:
    delay_rating = randint(2, 6)
  else:
    delay_rating = randint(0, 3)

  # calculate speed_rating
  wpm = i[1]
  if wpm <= 90:
    speed_rating = randint(0,4) # when its too slow to read
  elif 90 < wpm <= 100:
    speed_rating = randint(5,8)
  elif 100 < wpm <= 120:
    speed_rating = randint(8,10)
  elif 120 < wpm <= 220:
    speed_rating = randint(5,8)
  else:
    speed_rating = randint(0,4)

  # calculate grammar errors AND verbatim_ness
  sentence_sim = i[2]
  if sentence_sim == 100:
    grammar_rating = 10
    verbatim_rating = 10
  elif 96 <= sentence_sim < 100:
    verbatim_rating = randint(8,10)
    grammar_rating = randint(0,10)
  elif 90 <= sentence_sim < 96:
    verbatim_rating = randint(4,7)
    grammar_rating = randint(0,10)
  else:
    verbatim_rating = randint(0,3)
    grammar_rating = randint(0,10)

  # calculate misspell_rating
  misspelled = i[3]
  if misspelled == 0:
    misspell_rating = 10
  elif 0 < misspelled <= 3:
    misspell_rating = randint(4,9)
  else:
    misspell_rating = randint(0,3)

  rating_list[0].append(speed_rating)
  rating_list[1].append(delay_rating)
  rating_list[2].append(misspell_rating)
  rating_list[3].append(grammar_rating)
  rating_list[4].append(verbatim_rating)
p = np.asarray(rating_list)

for i in p:
  c = np.column_stack((c, i))

print(c)
print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)



with open('generated_data.csv', 'w') as mf:
  wr = csv.writer(mf)
  for i in c:
    wr.writerow(i)
