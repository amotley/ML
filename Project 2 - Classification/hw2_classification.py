#python hw2_classification.py X_train.csv y_train.csv X_test.csv
from __future__ import division
import numpy as np
import sys
from operator import add
import math as math

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

def pluginClassifier():  
    count_0 = 0.0
    count_1 = 0.0
    count_2 = 0.0
    count_3 = 0.0
    count_4 = 0.0
    count_5 = 0.0
    count_6 = 0.0
    count_7 = 0.0
    count_8 = 0.0
    count_9 = 0.0
    x_0 = []
    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []
    x_5 = []
    x_6 = []
    x_7 = []
    x_8 = []
    x_9 = []
    totalCount = len(y_train)
    index = 0
    for y in y_train:
      if y == 0:
          count_0 = count_0 + 1.0
          x_0 = x_0 + [X_train[index]]
      if y == 1:
          count_1 = count_1 + 1.0
          x_1 = x_1 + [X_train[index]]
      if y == 2:
          count_2 = count_2 + 1.0
          x_2 = x_2 + [X_train[index]]
      if y == 3:
          count_3 = count_3 + 1.0
          x_3 = x_3 + [X_train[index]]
      if y == 4:
          count_4 = count_4 + 1.0
          x_4 = x_4 + [X_train[index]]
      if y == 5:
          count_5 = count_5 + 1.0
          x_5 = x_5 + [X_train[index]]
      if y == 6:
          count_6 = count_6 + 1.0
          x_6 = x_6 + [X_train[index]]
      if y == 7:
          count_7 = count_7 + 1.0
          x_7 = x_7 + [X_train[index]]
      if y == 8:
          count_8 = count_8 + 1.0
          x_8 = x_8 + [X_train[index]]
      if y == 9:
          count_9 = count_9 + 1.0
          x_9 = x_9 + [X_train[index]]
      index = index + 1

    #find pi
    pi_0 = count_0 / totalCount
    pi_1 = count_1 / totalCount
    pi_2 = count_2 / totalCount
    pi_3 = count_3 / totalCount
    pi_4 = count_4 / totalCount
    pi_5 = count_5 / totalCount
    pi_6 = count_6 / totalCount
    pi_7 = count_7 / totalCount
    pi_8 = count_8 / totalCount
    pi_9 = count_9 / totalCount

    pi = [pi_0, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9]

    #find mew
    sum_0 = x_0[0]*0
    for x in x_0:
      sum_0 = map(add, sum_0, x)
    mew_0 = 1.0/count_0 * np.asarray(sum_0)
    
    sum_1 = x_1[0]*0
    for x in x_1:
      sum_1 = map(add, sum_1, x)
    mew_1 = 1.0/count_1 * np.asarray(sum_1)

    sum_2 = x_2[0]*0
    for x in x_2:
      sum_2 = map(add, sum_2, x)
    mew_2 = 1.0/count_2 * np.asarray(sum_2)
    
    sum_3 = x_3[0]*0
    for x in x_3:
      sum_3 = map(add, sum_3, x)
    mew_3 = 1.0/count_3 * np.asarray(sum_3)

    sum_4 = x_4[0]*0
    for x in x_4:
      sum_4 = map(add, sum_4, x)
    mew_4 = 1.0/count_4 * np.asarray(sum_4)
    
    sum_5 = x_5[0]*0
    for x in x_5:
      sum_5 = map(add, sum_5, x)
    mew_5 = 1.0/count_5 * np.asarray(sum_5)

    sum_6 = x_6[0]*0
    for x in x_6:
      sum_6 = map(add, sum_6, x)
    mew_6 = 1.0/count_6 * np.asarray(sum_6)
    
    sum_7 = x_7[0]*0
    for x in x_7:
      sum_7 = map(add, sum_7, x)
    mew_7 = 1.0/count_7 * np.asarray(sum_7)

    sum_8 = x_8[0]*0
    for x in x_8:
      sum_8 = map(add, sum_8, x)
    mew_8 = 1.0/count_8 * np.asarray(sum_8)
    
    sum_9 = x_9[0]*0
    for x in x_9:
      sum_9 = map(add, sum_9, x)
    mew_9 = 1.0/count_9 * np.asarray(sum_9)


    #find sigma
    xMinusMew = np.subtract(np.asarray(x_0), np.asarray(mew_0))
    sigma_0 = 1.0/count_0 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_1), np.asarray(mew_1))
    sigma_1 = 1.0/count_1 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_2), np.asarray(mew_2))
    sigma_2 = 1.0/count_2 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_3), np.asarray(mew_3))
    sigma_3 = 1.0/count_3 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_4), np.asarray(mew_4))
    sigma_4 = 1.0/count_4 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_5), np.asarray(mew_5))
    sigma_5 = 1.0/count_5 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_6), np.asarray(mew_6))
    sigma_6 = 1.0/count_6 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_7), np.asarray(mew_7))
    sigma_7 = 1.0/count_7 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_8), np.asarray(mew_8))
    sigma_8 = 1.0/count_8 * np.matmul(xMinusMew.transpose(), xMinusMew)

    xMinusMew = np.subtract(np.asarray(x_9), np.asarray(mew_9))
    sigma_9 = 1.0/count_9 * np.matmul(xMinusMew.transpose(), xMinusMew)

    #find probabilities
    toReturn = []
    for x in X_test:
      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_0))
      exp_0 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_0)),xMinusMew)
      f_0 = pi_0 * np.linalg.det(sigma_0)**(-.5) * math.exp(exp_0)

      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_1))
      exp_1 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_1)),xMinusMew)
      f_1 = pi_1 * np.linalg.det(sigma_1)**(-.5) * math.exp(exp_1)

      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_2))
      exp_2 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_2)),xMinusMew)
      f_2 = pi_2 * np.linalg.det(sigma_2)**(-.5) * math.exp(exp_2)
      
      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_3))
      exp_3 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_3)),xMinusMew)
      f_3 = pi_3 * np.linalg.det(sigma_3)**(-.5) * math.exp(exp_3)

      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_4))
      exp_4 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_4)),xMinusMew)
      f_4 = pi_4 * np.linalg.det(sigma_4)**(-.5) * math.exp(exp_4)
      
      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_5))
      exp_5 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_5)),xMinusMew)
      f_5 = pi_5 * np.linalg.det(sigma_5)**(-.5) * math.exp(exp_5)
      
      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_6))
      exp_6 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_6)),xMinusMew)
      f_6 = pi_6 * np.linalg.det(sigma_6)**(-.5) * math.exp(exp_6)
      
      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_7))
      exp_7 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_7)),xMinusMew)
      f_7 = pi_7 * np.linalg.det(sigma_7)**(-.5) * math.exp(exp_7)

      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_8))
      exp_8 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_8)),xMinusMew)
      f_8 = pi_8 * np.linalg.det(sigma_8)**(-.5) * math.exp(exp_8)
      
      xMinusMew = np.subtract(np.asarray(x), np.asarray(mew_9))
      exp_9 = (-1.0/2.0)*np.matmul(np.matmul(xMinusMew.transpose(), np.linalg.inv(sigma_9)),xMinusMew)
      f_9 = pi_9 * np.linalg.det(sigma_9)**(-.5) * math.exp(exp_9)

      f_sum = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + f_9
      toReturn = toReturn + [[f_0/f_sum,f_1/f_sum,f_2/f_sum,f_3/f_sum,f_4/f_sum,f_5/f_sum,f_6/f_sum,f_7/f_sum,f_8/f_sum,f_9/f_sum]]

    return toReturn

final_outputs = pluginClassifier() # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file

