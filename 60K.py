#!/usr/bin/python3

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

# Functions

# weighing function
def weighing(n, k, f):
    return (1 / (1 + np.exp(- (n - k) / f)))

# Handling High Cardinality Categorical Features
def ev_cat(n, ev_loc, ev_glob):
    k = 100
    f = 100
    lamb = weighing(n, k, f)
    return (lamb * ev_loc + (1 - lamb) * ev_glob)

def calculate_cat(df):
    return ev_cat(df["size"], df["mean"], ev_price)

# Read and clean
df_merc = pd.read_table('./train.tsv', index_col = 0)
df_merc.drop(df_merc[df_merc.price == 0].index, inplace = True)
df_cats = df_merc.category_name.str.split('/')
cats = ['first_cat', 'second_cat', 'third_cat']

for i in range(3):
    df_merc[cats[i]] = df_cats.str.get(i)

df_merc.drop(columns = ['category_name', 'name', 'item_description'], inplace = True)

# Hard clean
df_merc.dropna(axis=0, how='any', inplace=True)

# Mapping Cat to their EV
ev_and_count_by_brand = df_merc.groupby('brand_name').price.aggregate(['size', 'mean'])
ev_and_count_by_fcat = df_merc.groupby('first_cat').price.aggregate(['size', 'mean'])
ev_and_count_by_scat = df_merc.groupby('second_cat').price.aggregate(['size', 'mean'])
ev_and_count_by_tcat = df_merc.groupby('third_cat').price.aggregate(['size', 'mean'])
ev_price = df_merc.price.mean()
brand_cortable = ev_and_count_by_brand.apply(calculate_cat, axis = 1)
fcat_cortable = ev_and_count_by_fcat.apply(calculate_cat, axis = 1)
scat_cortable = ev_and_count_by_scat.apply(calculate_cat, axis = 1)
tcat_cortable = ev_and_count_by_tcat.apply(calculate_cat, axis = 1)
df_merc.brand_name.replace(brand_cortable, inplace = True)
df_merc.first_cat.replace(fcat_cortable, inplace = True)
df_merc.second_cat.replace(scat_cortable, inplace = True)
df_merc.third_cat.replace(tcat_cortable, inplace = True)

# first regression
#estimator = RandomForestRegressor(random_state=0, n_estimators=20, n_jobs=-1, verbose=1)
#score = cross_val_score(estimator, df_merc.drop(columns = 'price'), df_merc.price, n_jobs=-1)
#print(score)
#
#estimator.fit(df_merc.drop(columns = 'price'), df_merc.price)
#df_merc['predicted'] = estimator.predict(df_merc.drop(columns = 'price'))
#df_merc['eval'] = np.power(np.log(df_merc['predicted'] + 1) - np.log(df_merc['price'] + 1), 2)
#eval1 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
#print(eval1)

# 2nd
"""estimator2 = ExtraTreesRegressor(random_state=0, n_estimators=20, n_jobs=-1, verbose=1)
score2 = cross_val_score(estimator2, df_merc_cat.drop(columns = 'price'), df_merc_cat.price, n_jobs=-1)
print(score2)

estimator2.fit(df_merc_cat.drop(columns = 'price'), df_merc_cat.price)
df_merc_cat['predicted2'] = estimator2.predict(df_merc_cat.drop(columns = 'price'))
df_merc_cat['eval2'] = np.power(np.log(df_merc_cat['predicted2'] + 1) - np.log(df_merc_cat['price'] + 1), 2)
eval2 = np.sqrt(1 / len(df_merc_cat['eval2']) * df_merc_cat['eval2'].sum())
print(eval2)"""

# 3rd
"""estimator3 = Ridge()
score = cross_val_score(estimator3, df_merc_cat.drop(columns = 'price'), df_merc_cat.price, n_jobs=-1)
print(score)
estimator3.fit(df_merc_cat.drop(columns = 'price'), df_merc_cat.price)
df_merc_cat['predicted3'] = estimator3.predict(df_merc_cat.drop(columns = 'price'))
print(df_merc_cat['predicted3'])
df_merc_cat['eval3'] = np.power(np.log(df_merc_cat['predicted3'] + 1) - np.log(df_merc_cat['price'] + 1), 2)
eval3 = np.sqrt(1 / len(df_merc_cat['eval3']) * df_merc_cat['eval3'].sum())
print(eval3)"""
