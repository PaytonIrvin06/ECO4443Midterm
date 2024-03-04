# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:25:49 2024

@author: Payton Irvin
"""


from itertools import combinations
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from statsmodels.api import add_constant
import numpy as np
import statsmodels.api as sm
from numpy.random import seed
from math import log
from math import tan
from math import sqrt

import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

# Payton Desktop Filepath
data = read_csv('C:/Users/Payton Irvin/Documents/UCF/ECO4443/Python/Data/mid_term_dataset.csv')


# Creates a heatmapped correlation matrix to get a feel for the data
corr = data.corr()

sns.heatmap(corr);
heatmap = sns.heatmap(data.corr().round(1), annot=True)
heatmap.set_title("Correlation Heatmap")
print(heatmap)


#scaling the dependent variable
data['price'] = data['price']/1000


# Creating variables to evaluate in competing models

#data['parcel_home_interaction'] = data.parcel_size*data.home_size
data['parcel_home_ratio'] = data.home_size/data.parcel_size
data['bath_bed_ratio'] = data.baths/data.beds
data['r_coord'] = sqrt(data.x_coord**2) + (data.y_coord**2)


#data['x_y_interaction'] = data['x_coord']*data['y_coord']
#data['area_sqd'] = data.home_size**2
#data['area_cubed'] = data.home_size**3
#data['parcel_sqd'] = data.parcel_size**2
#data['parcel_cubed'] = data.parcel_size**3
#data['age_sqd'] = data.age**2
#data['cbd_lakes_interaction'] = data.dist_cbd*data.dist_lakes
#data['cbd_sqd'] = data.dist_cbd**2
#data['lakes_sqd'] = data.dist_lakes**2
#data['bed_bath_interaction'] = data.beds*data.baths
#data['bed_home_size_interaction'] = data.beds*data.home_size
#data['bath_home_size_interaction'] = data.home_size*data.baths
#data['bed_bath_home_size_interaction'] = data.home_size*data.baths*data.beds
#data['pool_home_size_interaction'] = data.home_size*data.pool

# Creating dummies for year
data['year_2000'] = 0
data.year_2000[data['year'] == 2000] = 1
data['year_2001'] = 0
data.year_2001[data['year'] == 2001] = 1
data['year_2002'] = 0
data.year_2002[data['year'] == 2002] = 1
data['year_2003'] = 0
data.year_2003[data['year'] == 2003] = 1
data['year_2004'] = 0
data.year_2004[data['year'] == 2004] = 1
data['year_2005'] = 0
data.year_2005[data['year'] == 2005] = 1
        
# dummies for beds
        
data['bed_1'] = 0
data.bed_1[data['beds'] == 1] = 1
data['bed_2'] = 0
data.bed_2[data['beds'] == 2] = 1
data['bed_3'] = 0
data.bed_3[data['beds'] == 3] = 1
data['bed_4'] = 0
data.bed_4[data['beds'] == 4] = 1
data['bed_5'] = 0
data.bed_5[data['beds'] == 5] = 1
data['bed_6'] = 0
data.bed_6[data['beds'] == 6] = 1
data['bed_7'] = 0
data.bed_7[data['beds'] == 7] = 1

#dummies for baths


data['bath_1']= 0
data.bath_1[data['baths'] == 1] = 1
data['bath_15']= 0
data.bath_15[data['baths'] == 1.5] = 1
data['bath_2']= 0
data.bath_2[data['baths'] == 2] = 1
data['bath_25']= 0
data.bath_25[data['baths'] == 2.5] = 1
data['bath_3']= 0
data.bath_3[data['baths'] == 3] = 1
data['bath_35']= 0
data.bath_35[data['baths'] == 3.5] = 1
data['bath_4']= 0
data.bath_4[data['baths'] == 4] = 1
data['bath_45']= 0
data.bath_45[data['baths'] == 4.5] = 1
data['bath_5']= 0
data.bath_5[data['baths'] == 5] = 1
data['bath_55']= 0
data.bath_55[data['baths'] == 5.5] = 1
data['bath_6']= 0
data.bath_6[data['baths'] == 6] = 1
data['bath_65']= 0
data.bath_65[data['baths'] == 6.5] = 1
data['bath_7']= 0
data.bath_7[data['baths'] == 7] = 1
data['bath_75']= 0
data.bath_75[data['baths'] == 7.5] = 1
data['bath_8']= 0
data.bath_8[data['baths'] == 8] = 1
data['bath_9']= 0
data.bath_9[data['baths'] == 9] = 1



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

################################################################################
#Trying a new model using polynomial features


seed(1234)
data = data.sample(len(data))

x_combos = []
for n in range(1, 21):
    combos = combinations(['age', 'beds', 'baths', 'home_size', 'parcel_size', 'pool', 'bath_bed_ratio',\
                           'bed_3', 'bed_5', 'bath_2', 'bath_3', 'bath_4', 'bath_45', 'bath_5',\
                               'bath_55', 'bath_65', 'bath_75', 'x_coord', 'y_coord', 'parcel_home_ratio'], n)
    x_combos.extend(combos)

y = data['price']

mse = {}

for n in range(0, len(x_combos)):
    for j in range(1, 3): 
        combo_list = list(x_combos[n])
        x = data[combo_list]
        poly = PolynomialFeatures(j)
        poly_x = poly.fit_transform(x)
        model = LinearRegression()
        cv_scores = cross_validate(model, poly_x, y, cv=10, scoring=('neg_mean_squared_error'))
        mse[str(combo_list), j] = np.mean(cv_scores['test_score'])

print("Outcomes from the Best Linear Regression Model:")
min_mse = abs(max(mse.values()))
print("Minimum Average Test MSE:", min_mse.round(2))
for possibles, i in mse.items():
    if i == -min_mse:
        print("The Combination of Variables:", possibles)


##############################################################################
#Model 2
x = data[['home_size', 'pool', 'year', 'age', 'parcel_home_ratio', 'dist_lakes', 'dist_cbd', 'x_coord', 'y_coord', 'bath_bed_ratio', 'bed_3']]
poly = PolynomialFeatures(3)
poly_x = poly.fit_transform(x)

# Add variable names
x = DataFrame(poly.fit_transform(x), columns=poly.get_feature_names_out(x.columns))
y = data['price']

best_model = sm.OLS(y, x)
results = best_model.fit()
print(results.summary())

pred = results.predict(poly_x)
mse_best_model2 = sum((data.price - pred)**2)/results.nobs
mse_best_model2