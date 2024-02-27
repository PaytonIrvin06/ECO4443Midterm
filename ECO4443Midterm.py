# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:25:49 2024

@author: Payton Irvin
"""


from itertools import combinations
from pandas import read_csv
from statsmodels.api import add_constant
import numpy as np
import statsmodels.api as sm
from numpy.random import seed
from math import log


import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression

# Payton Desktop Filepath
data = read_csv('C:/Users/Payton Irvin/Documents/UCF/ECO4443/Python/Data/mid_term_dataset.csv')


# Creates a correlation matrix to get a feel for the data
corr = data.corr()

#scaling the dependent variable
data['price'] = data['price']/1000


# Creating variables to evaluate in competing models

data['parcel_home_interaction'] = data.parcel_size*data.home_size
data['parcel_home_ratio'] = data.parcel_size/data.home_size


data['x_y_interaction'] = data['x_coord']*data['y_coord']

        
data['area_sqd'] = data.home_size**2
data['area_cubed'] = data.home_size**3

data['parcel_sqd'] = data.parcel_size**2
data['parcel_cubed'] = data.parcel_size**3


data['age_sqd'] = data.age**2

data['cbd_lakes_interaction'] = data.dist_cbd*data.dist_lakes
data['cbd_sqd'] = data.dist_cbd**2

data['lakes_sqd'] = data.dist_lakes**2

data['bed_bath_interaction'] = data.beds*data.baths
data['bed_home_size_interaction'] = data.beds*data.home_size
data['bath_home_size_interaction'] = data.home_size*data.baths
data['bed_bath_home_size_interaction'] = data.home_size*data.baths*data.beds

data['pool_home_size_interaction'] = data.home_size*data.pool

# Creating dummies for year
data['year_2000'] = 0
data['year_2001'] = 0
data['year_2002'] = 0
data['year_2003'] = 0
data['year_2004'] = 0
data['year_2005'] = 0

for i in range(0, len(data.year)):
    if (data.year[i] == 2000):
        data.year_2000[i] = 1
    elif (data.year[i] == 2001):
        data.year_2001[i] = 1
    elif (data.year[i] == 2002):
        data.year_2002[i] = 1
    elif (data.year[i] == 2003):
        data.year_2003[i] = 1
    elif (data.year[i] == 2004):
        data.year_2004[i] = 1
    elif (data.year[i] == 2005):
        data.year_2005[i] = 1
        
# dummies for beds
        
data['bed_1'] = 0
data['bed_2'] = 0
data['bed_3'] = 0
data['bed_4'] = 0
data['bed_5'] = 0
data['bed_6'] = 0
data['bed_7'] = 0
data['bed_8'] = 0

for i in range(0, len(data.beds)):
    if (data.beds[i] == 1):
        data.bed_1[i] = 1
    elif (data.beds[i] == 2):
        data.bed_2[i] = 1
    elif (data.beds[i] == 3):
        data.bed_3[i] = 1
    elif (data.beds[i] == 4):
        data.bed_4[i] = 1
    elif (data.beds[i] == 5):
        data.bed_5[i] = 1
    elif (data.beds[i] == 6):
        data.bed_6[i] = 1
    elif (data.beds[i] == 7):
        data.bed_7[i] = 1
    elif (data.beds[i] == 8):
        data.bed_8[i] = 1

#dummies for baths

data['bath_1'] = 0
data['bath_15'] = 0
data['bath_2'] = 0
data['bath_25'] = 0
data['bath_3'] = 0
data['bath_35'] = 0
data['bath_4'] = 0
data['bath_45'] = 0
data['bath_5'] = 0
data['bath_55'] = 0
data['bath_6'] = 0
data['bath_65'] = 0
data['bath_7'] = 0
data['bath_75'] = 0
data['bath_8'] = 0
data['bath_9'] = 0


for i in range(0, len(data.baths)):
    if (data.baths[i] == 1):
        data.bath_1[i] = 1
    elif (data.baths[i] == 1.5):
        data.bath_15[i] = 1
    elif (data.baths[i] == 2):
        data.bath_2[i] = 1
    elif (data.baths[i] == 2.5):
        data.bath_25[i] = 1
    elif (data.baths[i] == 3):
        data.bath_3[i] = 1
    elif (data.baths[i] == 3.5):
        data.bath_35[i] = 1
    elif (data.baths[i] == 4):
        data.bath_4[i] = 1
    elif (data.baths[i] == 4.5):
        data.bath_45[i] = 1
    elif (data.baths[i] == 5):
        data.bath_5[i] = 1
    elif (data.baths[i] == 5.5):
        data.bath_55[i] = 1
    elif (data.baths[i] == 6):
        data.bath_6[i] = 1
    elif (data.baths[i] == 6.5):
        data.bath_65[i] = 1
    elif (data.baths[i] == 7):
        data.bath_7[i] = 1
    elif (data.baths[i] == 7.5):
        data.bath_75[i] = 1
    elif (data.baths[i] == 8):
        data.bath_8[i] = 1
    elif (data.baths[i] == 9):
        data.bath_9[i] = 1


#creating model variable combinations

seed(1234)
data = data.sample(len(data))

x_combos = []
for n in range(1,58):
    combos = combinations(['year', 'baths', 'age', 'beds', 'home_size', 'parcel_size',\
                           'pool', 'dist_cbd', 'dist_lakes', 'x_coord', 'y_coord', 'parcel_home_interaction',\
                               'parcel_home_ratio', 'x_y_interaction', 'area_sqd', 'area_cubed', 'parcel_sqd',\
                                   'parcel_cubed', 'age_sqd', 'cbd_lakes_interaction', 'cbd_sqd', 'lakes_sqd',\
                                       'bed_bath_interaction', 'bed_home_size_interaction', 'bath_home_size_interaction',\
                                           'bed_bath_home_size_interaction', 'pool_home_size_interaction',\
                                               'year_2000', 'year_2001', 'year_2002', 'year_2003', 'year_2004',\
                                                   'year_2005', 'bed_1', 'bed_2', 'bed_3', 'bed_4', 'bed_5', 'bed_6',\
                                                       'bed_7', 'bed_8', 'bath_1', 'bath_15', 'bath_2', 'bath_25', 'bath_3',\
                                                           'bath_35', 'bath_4', 'bath_45', 'bath_5', 'bath_55', 'bath_6', 'bath_65',\
                                                               'bath_7', 'bath_75', 'bath_8', 'bath_9'], n)
    x_combos.extend(combos)

y = data['price']

# A dictionary is created to store the variable combinations
# and associated average (over folds) test mse values

mse = {}


for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    x = data[combo_list]
    model = LinearRegression()
    cv_scores = cross_validate(model, x, y, cv=10, scoring=('neg_mean_squared_error'))
    mse[str(combo_list)] = np.mean(cv_scores['test_score'])

print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse.round(3))
for possibles, i in mse.items():
    if i == -min_mse:
        print("The Combination of Variables:", possibles)
