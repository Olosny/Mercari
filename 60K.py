#!/usr/bin/python3

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

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

def calculate_cat(df):
    return ev_cat(df["size"], df["mean"], ev_price)

# Read and clean
df_merc = pd.read_table('./train.tsv', index_col = 0)
df_merc.drop(df_merc[df_merc.price == 0].index, inplace = True)
df_cats = df_merc.category_name.str.split('/')
cats = ['first_cat', 'second_cat', 'third_cat']

for i in range(3):
    df_merc[cats[i]] = df_cats.str.get(i)


# Hard clean
df_merc.dropna(axis=0, how='any', inplace=True)

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

"""
# Mapping Cat to their EV
ev_price = df_merc.price.mean()
ev_cat1 = df_merc.groupby('cat1').price.aggregate(['size', 'mean'])
ev_cat12 = df_merc.groupby('cat12').price.aggregate(['size', 'mean'])
ev_cat123 = df_merc.groupby('cat123').price.aggregate(['size', 'mean'])
ev_cat1234 = df_merc.groupby('cat1234').price.aggregate(['size', 'mean'])

cat1 = ev_cat1.apply(calculate_cat, axis = 1)

brand_cortable = ev_and_count_by_brand.apply(calculate_cat, axis = 1)
fcat_cortable = ev_and_count_by_fcat.apply(calculate_cat, axis = 1)
scat_cortable = ev_and_count_by_scat.apply(calculate_cat, axis = 1)
tcat_cortable = ev_and_count_by_tcat.apply(calculate_cat, axis = 1)
df_merc.brand_name.replace(brand_cortable, inplace = True)
df_merc.first_cat.replace(fcat_cortable, inplace = True)
df_merc.second_cat.replace(scat_cortable, inplace = True)
df_merc.third_cat.replace(tcat_cortable, inplace = True)"""

# first regression
estimator = RandomForestRegressor(random_state=0, n_estimators=20, n_jobs=-1, verbose=1)
#score = cross_val_score(estimator, df_merc.drop(columns = 'price'), df_merc.price, n_jobs=-1)
#print(score)
#
estimator.fit(df_merc.drop(columns = 'price'), df_merc.price)
df_merc['predicted'] = estimator.predict(df_merc.drop(columns = 'price'))
df_merc['eval'] = np.power(np.log(df_merc['predicted'] + 1) - np.log(df_merc['price'] + 1), 2)
eval1 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
print(eval1)
df_merc.drop(columns=['predicted','eval'], inplace=True)

# 2nd
estimator2 = ExtraTreesRegressor(random_state=0, n_estimators=20, n_jobs=-1, verbose=1)
#score2 = cross_val_score(estimator2, df_merc_cat.drop(columns = 'price'), df_merc_cat.price, n_jobs=-1)
#print(score2)

estimator2.fit(df_merc.drop(columns = 'price'), df_merc.price)
df_merc['predicted'] = estimator2.predict(df_merc.drop(columns = 'price'))
df_merc['eval'] = np.power(np.log(df_merc['predicted'] + 1) - np.log(df_merc['price'] + 1), 2)
eval2 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
print(eval2)
df_merc.drop(columns=['predicted','eval'], inplace=True)

# 3rd
estimator3 = Ridge()
#score = cross_val_score(estimator3, df_merc_cat.drop(columns = 'price'), df_merc_cat.price, n_jobs=-1)
#print(score)
estimator3.fit(df_merc.drop(columns = 'price'), df_merc.price)
df_merc['predicted'] = estimator3.predict(df_merc.drop(columns = 'price'))
df_merc['predicted'][df_merc['predicted']<3]=3 
df_merc['eval'] = np.power(np.log(df_merc['predicted'] + 1) - np.log(df_merc['price'] + 1), 2)
eval3 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
print(eval3)
df_merc.drop(columns=['predicted','eval'], inplace=True)

# 4th
estimator4 = SGDRegressor()
#score = cross_val_score(estimator4, df_merc_cat.drop(columns = 'price'), df_merc_cat.price, n_jobs=-1)
#print(score)
estimator4.fit(df_merc.drop(columns = 'price'), df_merc.price)
df_merc['predicted'] = estimator4.predict(df_merc.drop(columns = 'price'))
df_merc['predicted'][df_merc['predicted']<3]=3 
df_merc['eval'] = np.power(np.log(df_merc['predicted'] + 1) - np.log(df_merc['price'] + 1), 2)
eval4 = np.sqrt(1 / len(df_merc['eval']) * df_merc['eval'].sum())
print(eval4)
df_merc.drop(columns=['predicted','eval'], inplace=True)
