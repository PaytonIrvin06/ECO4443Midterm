# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:25:49 2024

@author: Payton Irvin
"""

import pandas as pd
from itertools import combinations
from pandas import read_csv
from pandas import DataFrame
import numpy as np
import statsmodels.api as sm
from numpy.random import seed
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso

# Payton's Desktop File Path

data = read_csv('C:/Users/Payton Irvin/Documents/UCF/ECO4443/Python/Data/mid_term_dataset.csv')

# Payton's Surface File Path

#data = read_csv('C:/UCF/ECO4443/Python/Data/mid_term_dataset.csv')

# Generating summary statistics for existing variables:

stats.describe(data.price)

stats.describe(data.year)

stats.describe(data.age)

stats.describe(data.beds)

stats.describe(data.baths)

stats.describe(data.home_size)

stats.describe(data.parcel_size)

stats.describe(data.pool)

stats.describe(data.dist_cbd)

stats.describe(data.dist_lakes)

stats.describe(data.x_coord)

stats.describe(data.y_coord)


# Creating Histograms for existing variables


plt.hist(data.price, bins=100, color='#22878a')

plt.xlabel("Sales Price")

plt.ylabel("# of observations")

plt.title("Sales Price")

plt.show()


plt.hist(data.year, bins=100, color='#5E33FF')

plt.xlabel("Year of Sale")

plt.ylabel("# of observations")

plt.title("Year of Sale")

plt.show()


plt.hist(data.age, bins=100, color='#FF334B')

plt.xlabel("Age of Home")

plt.ylabel("# of Observations")

plt.title("Age of Home")

plt.show()



plt.hist(data.beds, bins=100, color='#22878a')

plt.xlabel("Bedrooms")

plt.ylabel("# of Observations")

plt.title("Number of Bedrooms")

plt.show()


plt.hist(data.baths, bins=100, color='#5E33FF')

plt.xlabel("Bathrooms")

plt.ylabel("# of Observations")

plt.title("Number of Bathrooms")

plt.show()


plt.hist(data.home_size, bins=100, color='#FF334B')

plt.xlabel("Home Size (Sq ft)")

plt.ylabel("# of Observations")

plt.title("Size of Home")

plt.show()


plt.hist(data.parcel_size, bins=100, color='#22878a')

plt.xlabel("Parcel Size (Sq ft)")

plt.ylabel("# of Observations")

plt.title("Size of Parcel")

plt.show()


plt.hist(data.pool, bins=100, color='#5E33FF')

plt.xlabel("Pool")

plt.ylabel("# of Observations")

plt.title("Pool Homes")

plt.show()


plt.hist(data.dist_cbd, bins=100, color='#FF334B')

plt.xlabel("Distance from CBD (m)")

plt.ylabel("# of Observations")

plt.title("Distance from CBD")

plt.show()


plt.hist(data.dist_lakes, bins=100, color='#22878a')

plt.xlabel("Distance from Nearest Lake (m)")

plt.ylabel("# of Observations")

plt.title("Distance from Nearest Lake")

plt.show()


plt.hist(data.x_coord, bins=100, color='#5E33FF')

plt.xlabel("X coordinate")

plt.ylabel("# of Observations")

plt.title("X Coordinate of Home")

plt.show()


plt.hist(data.y_coord, bins=100, color='#FF334B')

plt.xlabel("Y coordinate")

plt.ylabel("# of Observations")

plt.title("Y Coordinate of Home")

plt.show()


# Creates a heatmapped correlation matrix to get a feel for the data

corr = data.corr()

sns.heatmap(corr)

heatmap = sns.heatmap(data.corr().round(1), annot=True)

heatmap.set_title("Correlation Heatmap")

print(heatmap)



# Creating scatterplots for each variable with respect to price


plt.scatter(data.year,data.price, color='#5E33FF')

plt.xlabel("Year Sold")

plt.ylabel("Sales Price")

plt.title("Year Sold")

plt.show()



plt.scatter(data.age,data.price, color='#FF334B')

plt.xlabel("Age of Home")

plt.ylabel("Sales Price")

plt.title("Age of Home")

plt.show()



plt.scatter(data.beds,data.price, color='#22878a')

plt.xlabel("Number of Bedrooms")

plt.ylabel("Sales Price")

plt.title("Number of Bedrooms")

plt.show()



plt.scatter(data.baths,data.price, color='#5E33FF')

plt.xlabel("Number of Bathrooms")

plt.ylabel("Sales Price")

plt.title("Number of Bathrooms")

plt.show()


plt.scatter(data.home_size,data.price, color='#FF334B')

plt.xlabel("Size of Home (Sq ft)")

plt.ylabel("Sales Price")

plt.title("Size of Home")

plt.show()


plt.scatter(data.parcel_size,data.price, color='#22878a')

plt.xlabel("Size of Parcel (Sq ft)")

plt.ylabel("Sales Price")

plt.title("Size of Parcel")

plt.show()


plt.scatter(data.pool,data.price, color='#5E33FF')

plt.xlabel("Pool")

plt.ylabel("Sales Price")

plt.title("Pool Home")

plt.show()


plt.scatter(data.dist_cbd,data.price, color='#FF334B')

plt.xlabel("Distance from CBD (m)")

plt.ylabel("Sales Price")

plt.title("Distance from CBD")

plt.show()


plt.scatter(data.dist_lakes,data.price, color='#22878a')

plt.xlabel("Distance from Nearest Lake (m)")

plt.ylabel("Sales Price")

plt.title("Distance from Nearest Lake")

plt.show()


plt.scatter(data.x_coord,data.price, color='#5E33FF')

plt.xlabel("X Coordinate")

plt.ylabel("Sales Price")

plt.title("X Coordinate of Home")

plt.show()


plt.scatter(data.y_coord,data.price, color='#FF334B')

plt.xlabel("Y Coordinate")

plt.ylabel("Sales Price")

plt.title("Y Coordinate of Home")

plt.show()


#scaling the dependent variable
data['price'] = data['price']/1000


# Creating variables to evaluate in competing models

data['parcel_home_ratio'] = data.home_size/data.parcel_size

data['bath_bed_ratio'] = data.baths/data.beds

# Creating Scatterplots for the new variables


plt.scatter(data.parcel_home_ratio,data.price, color='#22878a')

plt.xlabel("Parcel-Home Ratio")

plt.ylabel("Sales Price")

plt.title("Parcel-Home Ratio")

plt.show()



plt.scatter(data.bath_bed_ratio,data.price, color='#5E33FF')

plt.xlabel("Bath-Bed Ratio")

plt.ylabel("Sales Price")

plt.title("Ratio of Bathrooms per Bedroom of Homes")

plt.show()



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



#attempting a model with every possible combination of variables

seed(1234)
data = data.sample(len(data))

x_combos = []
for n in range(1,58):
    combos = combinations(['year', 'baths', 'age', 'beds', 'home_size', 'parcel_size',\
                           'pool', 'dist_cbd', 'dist_lakes', 'x_coord', 'y_coord', 'parcel_home_ratio',\
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
        
#Warning: This code takes forever to run
#Outcomes from the Best Linear Regression Model:
#Minimum Average Test MSE: Around 4900 at the time of cancelation

################################################################################
#Trying a new model using polynomial features and variables with correlation 
# coefficients > 0.2, and x and y coordinates.


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
    for j in range(2, 4): 
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

        
        
# New model using iterating accross a range of polynomial features values, 
# using only variables with correlation coeff > 0.3, as well as x, y, and the parcel home ratio
# excluding dummies


seed(1234)
data = data.sample(len(data))

x_combos = []
for n in range(1, 9):
    combos = combinations(['beds', 'baths', 'home_size', 'pool', 'bath_bed_ratio',\
                               'x_coord', 'y_coord', 'parcel_home_ratio'], n)
    x_combos.extend(combos)

y = data['price']

mse = {}

for n in range(0, len(x_combos)):
    for j in range(1, 4): 
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

# Outcome:
# Minimum Average Test MSE: 5047.64
# The Combination of Variables: ("['baths', 'home_size', 'pool', 'bath_bed_ratio', 'x_coord', 'y_coord', 'parcel_home_ratio']", 3)

###############################################################################

# New model using iterating accross a range of polynomial features values, 
# using only variables with correlation coeff > 0.3, as well as x, y, and the parcel home ratio
# including dummies


seed(1234)
data = data.sample(len(data))

x_combos = []
for n in range(1, 10):
    combos = combinations(['beds', 'baths', 'home_size', 'pool', 'bath_bed_ratio',\
                               'x_coord', 'y_coord', 'parcel_home_ratio', 'bed_5'], n)
    x_combos.extend(combos)

y = data['price']

mse = {}

for n in range(0, len(x_combos)):
    for j in range(1, 4): 
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

# Outcomes from the Best Linear Regression Model:
#Minimum Average Test MSE: 5154.06
#The Combination of Variables: ("['beds', 'home_size', 'pool', 'bath_bed_ratio', 'x_coord', 'y_coord', 'parcel_home_ratio']", 2)

###############################################################################

seed(1234)
data = data.sample(len(data))

x_combos = []
for n in range(1, 12):
    combos = combinations(['home_size', 'pool', 'year', 'age', 'parcel_home_ratio',\
                           'dist_lakes', 'dist_cbd', 'x_coord', 'y_coord', 'bath_bed_ratio', 'bed_3'], n)
    x_combos.extend(combos)

y = data['price']

mse = {}

for n in range(0, len(x_combos)):
    for j in range(1, 7): 
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
        

# Outcomes from the Best Linear Regression Model:
# Minimum Average Test MSE: 3412.17
# The Combination of Variables: ("['home_size', 'pool', 'year', 'age', 'parcel_home_ratio', 'dist_lakes', 'dist_cbd', 'x_coord', 'y_coord', 'bed_3']", 4)
###############################################################################

# Attempting a Lasso Model


seed(1234)
data = data.sample(len(data))

y = data['price']/1000
x = data[['home_size', 'pool', 'year', 'age', 'parcel_home_ratio', 'dist_lakes',\
          'dist_cbd', 'x_coord', 'y_coord', 'bath_bed_ratio', 'bed_3']]


x_scaled = preprocessing.scale(x)
x_scaled = pd.DataFrame(x_scaled, columns=('home_size', 'pool', 'year', 'age',\
                                           'parcel_home_ratio', 'dist_lakes', 'dist_cbd',\
                                               'x_coord', 'y_coord', 'bath_bed_ratio', 'bed_3'))

x_combos = []
for n in range(1,12):
    combos = combinations(['home_size', 'pool', 'year', 'age', 'parcel_home_ratio',\
                           'dist_lakes', 'dist_cbd', 'x_coord', 'y_coord', 'bath_bed_ratio', 'bed_3'], n)
    x_combos.extend(combos)


ols_mse = {}
lasso_mse = {}

for n in range(0, len(x_combos)):
    for j in range(5, 30, 5): 
        combo_list = list(x_combos[n])
        x = x_scaled[combo_list]

        lasso_cv_scores = cross_validate(Lasso(alpha=j), x, y, cv=10, scoring=('neg_mean_squared_error'))

        lasso_mse[str(combo_list), j] = np.mean(lasso_cv_scores['test_score'])

print("Outcomes from the Best Lasso Model:")
lasso_min_mse = abs(max(lasso_mse.values()))
print("Minimum Average Lasso Test MSE:", lasso_min_mse.round(3))
for possibles, r in lasso_mse.items():
    if r == -lasso_min_mse:
        print("The Lasso Combination of Variables:", possibles)
        
# For some reason, this model only outputs MSE's or 0.02 so either it doesn't work
# or is a VERY impressive predictive model.

###############################################################################

# Re-estimating the best model with the whole dataset


x = data[['home_size', 'pool', 'year', 'age', 'parcel_home_ratio', 'dist_lakes', 'dist_cbd', 'x_coord', 'y_coord', 'bed_3']]
    

poly = PolynomialFeatures(4)

poly_x = poly.fit_transform(x)



# Add variable names

x = DataFrame(poly.fit_transform(x), columns=poly.get_feature_names_out(x.columns))

y = list(data['price'])



best_model = sm.OLS(y, x)

results = best_model.fit()

# Print summary to a .txt file:
# (in case the summary is too long like if polynomial features > 3)
# Be sure to set your working directory if using this option.

with open("output.txt", "a") as f:
    
    print(results.summary(), file= f)

#Print summary to the console:

#print(results.summary())

pred = results.predict(poly_x)

mse_best_model = sum((data.price - pred)**2)/results.nobs

mse_best_model
# MSE = 2683.6381338397628
# R^2 = 0.869
##############################################################################

# Final Steps

from pandas import DataFrame

from pandas import read_csv

from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm


# Payton's File Path

#val_set = read_csv("C:/Users/Payton Irvin/Documents/UCF/ECO4443/Python/Data/mid_term_validation_set.csv")

# Playing with simulated validation sets

seed(3456)
data = data.sample(len(data))
val_set = data[:100]

# Creating the necessary variables for the validation set

val_set['parcel_home_ratio'] = val_set.home_size/val_set.parcel_size

val_set['bed_3'] = 0
val_set.bed_3[val_set['beds'] == 3] = 1

# Estimating the model

val_x = val_set[['home_size', 'pool', 'year', 'age', 'parcel_home_ratio', 'dist_lakes',\
                 'dist_cbd', 'x_coord', 'y_coord', 'bed_3']]

poly_val = PolynomialFeatures(4)

poly_val_x = poly_val.fit_transform(val_x)



val_x = DataFrame(poly_val.fit_transform(val_x), columns=poly_val.get_feature_names_out(val_x.columns))

val_y = val_set['price']



pred = results.predict(poly_val_x)

mse_best_model = sum((val_set.price - pred)**2)/len(val_set)

mse_best_model


###############################################################################
