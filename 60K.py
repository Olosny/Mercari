#!/usr/bin/python3

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

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

# Group low frequent brand_name
brand_counts = df_merc['brand_name'].value_counts().to_frame()
df_merc['brand_name'][df_merc['brand_name'].isin(brand_counts[brand_counts['brand_name']<10].index)] = 'Rare_brands'

# categorical to numeric
df_merc_cat = df_merc
cols = ['brand_name','first_cat','second_cat','third_cat']
for col in cols:
    df_merc[col] = pd.Categorical(df_merc[col])
    df_merc_cat[col] = df_merc[col].cat.codes


df_dummies = pd.get_dummies(df_merc, columns=['brand_name','first_cat','second_cat','third_cat'])


# first regression
estimator = RandomForestRegressor(random_state=0, n_estimators=20, n_jobs=-1, verbose=1)
#score = cross_val_score(estimator, df_dummies.drop(columns = 'price'), df_dummies.price, n_jobs=-1)
#print(score)

estimator.fit(df_dummies.drop(columns = 'price'), df_dummies.price)
df_dummies['predicted'] = estimator.predict(df_dummies.drop(columns = 'price'))
df_dummies['eval'] = np.power(np.log(df_dummies['predicted'] + 1) - np.log(df_dummies['price'] + 1), 2)
eval1 = np.sqrt(1 / len(df_dummies['eval']) * df_dummies['eval'].sum())
print(eval1)

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
