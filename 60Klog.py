#!/usr/bin/python3

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

# Functions
# weighing function
def weighing(n, k, f):
    return (1 / (1 + np.exp(- (n - k) / f)))

# Handling High Cardinality Categorical Features
def ev_cat(n, ev_loc, ev_glob):
    k = 100
    f = 10
    lamb = weighing(n, k, f)
    return (lamb * ev_loc + (1 - lamb) * ev_glob)


# Read and clean
df_merc = pd.read_table('./train.tsv', index_col = 0)
df_merc.drop(df_merc[df_merc.price == 0].index, inplace = True)
df_cats = df_merc.category_name.str.split('/')
cats = ['first_cat', 'second_cat', 'third_cat']

for i in range(3):
    df_merc[cats[i]] = df_cats.str.get(i)


# Hard clean
df_merc.dropna(axis=0, how='any', inplace=True)

# Log
df_merc['price'] = np.log(df_merc['price']+1)

# lambda
ev_price = df_merc.price.mean()
df_merc['cat1_s']=df_merc.groupby('first_cat').price.transform('size')
df_merc['cat1_m']=df_merc.groupby('first_cat').price.transform('mean')
df_merc['cat1_S']=ev_cat(df_merc['cat1_s'],df_merc['cat1_m'],ev_price)
df_merc.drop(columns = ['cat1_s','cat1_m'], inplace = True)

df_merc['cat12_s']=df_merc.groupby(['first_cat','second_cat']).price.transform('size')
df_merc['cat12_m']=df_merc.groupby(['first_cat','second_cat']).price.transform('mean')
df_merc['cat12_S']=ev_cat(df_merc['cat12_s'],df_merc['cat12_m'],df_merc['cat1_S'])
df_merc.drop(columns = ['cat12_s','cat12_m'], inplace = True)

df_merc['cat123_s']=df_merc.groupby(['first_cat','second_cat','third_cat']).price.transform('size')
df_merc['cat123_m']=df_merc.groupby(['first_cat','second_cat','third_cat']).price.transform('mean')
df_merc['cat123_S']=ev_cat(df_merc['cat123_s'],df_merc['cat123_m'],df_merc['cat12_S'])
df_merc.drop(columns = ['cat123_s','cat123_m'], inplace = True)

df_merc['cat1234_s']=df_merc.groupby(['first_cat','second_cat','third_cat','brand_name']).price.transform('size')
df_merc['cat1234_m']=df_merc.groupby(['first_cat','second_cat','third_cat','brand_name']).price.transform('mean')
df_merc['cat1234_S']=ev_cat(df_merc['cat1234_s'],df_merc['cat1234_m'],df_merc['cat123_S'])
df_merc.drop(columns = ['cat1234_s','cat1234_m'], inplace = True)
df_merc.drop(columns = ['cat1_S','cat12_S','cat123_S'], inplace = True)

df_merc.drop(columns = ['category_name', 'name', 'item_description','first_cat','second_cat','third_cat','brand_name'], inplace = True)



# Regression
estimator = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators':[1,2],'criterion':['mse','mae']}
grid = GridSearchCV(estimator,parameters)
grid.fit(df_merc.drop(columns = 'price'), df_merc.price)

#score = cross_val_score(estimator, df_merc.drop(columns = 'price'), df_merc.price, n_jobs=-1)
#print(score)
#
#   estimator.fit(df_merc.drop(columns = 'price'), df_merc.price)
#   df_merc['predicted'] = estimator.predict(df_merc.drop(columns = 'price'))
#   df_merc['eval'] = np.power(df_merc['predicted'] - df_merc['price'], 2)
#   eval1 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
#   print(n," ",eval1)
#   df_merc.drop(columns=['predicted','eval'], inplace=True)

"""
for n in range(2,101):
    estimator = ExtraTreesRegressor(random_state=0, n_estimators=n, n_jobs=-1)
#score = cross_val_score(estimator, df_merc.drop(columns = 'price'), df_merc.price, n_jobs=-1)
#print(score)
#
    estimator.fit(df_merc.drop(columns = 'price'), df_merc.price)
    df_merc['predicted'] = estimator.predict(df_merc.drop(columns = 'price'))
    df_merc['eval'] = np.power(df_merc['predicted'] - df_merc['price'], 2)
    eval1 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
    print(n," ",eval1)
    df_merc.drop(columns=['predicted','eval'], inplace=True)
"""
