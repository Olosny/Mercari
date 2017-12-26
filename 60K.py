#!/usr/bin/python3

# Imports
import pandas as pd                                                                                                                                           
import numpy as np                                                                                                                                            
import matplotlib.pyplot as plt                                                                                                                               
from sklearn.model_selection import cross_val_score                                                                                                           
from sklearn.model_selection import train_test_split                                                                                                          
from sklearn.ensemble import RandomForestRegressor                                                                                                            
from sklearn.ensemble import ExtraTreesRegressor                                                                                                              
from sklearn.ensemble import AdaBoostRegressor                                                                                                                
from sklearn.ensemble import BaggingRegressor                                                                                                                 
from sklearn.ensemble import GradientBoostingRegressor                                                                                                        
from sklearn.linear_model import Ridge                                                                                                                        
from sklearn.linear_model import SGDRegressor                                                                                                                 
from sklearn.linear_model import Lasso                                                                                                                        
from sklearn.linear_model import ElasticNet                                                                                                                   
from sklearn.svm import SVR                                                                                                                                   
from sklearn.svm import NuSVR
from nltk.corpus import stopwords
from collections import Counter
import re

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

# Read and clean
df_merc = pd.read_table('./train.tsv', index_col = 0)
df_merc.drop(df_merc[df_merc.price == 0].index, inplace = True)
df_cats = df_merc.category_name.str.split('/')
cats = ['first_cat', 'second_cat', 'third_cat']

for i in range(3):
    df_merc[cats[i]] = df_cats.str.get(i)

# Hard clean
df_merc.dropna(axis=0, how='any', inplace=True)

# Take log of prices
df_merc['price'] = np.log(df_merc['price']+1)


# Preprocessing categories and brand_names
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


# Item Description
stop = set(stopwords.words('english'))
stop.add('i\'d')
stop.add('i\'m')
stop.add('i\'ll')
stop.add('yes')
stop.add('no')
stop.add(';)')
stop.add('***')
stop.add('**')
stop.add(':)')
stop.add('(:')
stop.add('(;')
stop.add(':-)')
stop.add('//')
fake_chars_regex = '(^[(-])|([-:,!.?)"]*$)'
counts = Counter([ re.sub(fake_chars_regex, '', i) for i in ' '.join(list(df_merc.item_description)).lower().split()
    if i not in stop and len(i) > 1 ])

word_dict = {}
for k, v in counts.items():
    if v > 5000:
        word_dict.update({k : v})

word_count = pd.DataFrame.from_dict(word_dict, orient ='index')
try:
    word_count.drop(index = '', inplace = True)
except ValueError:
    pass
df_merc["cat_desc"] = df_merc.item_description.apply(lambda x : [i in x for i in word_count.index])
df_merc[[i + '_desc' for i in list(word_count.index)]] = pd.DataFrame(df_merc.cat_desc.values.tolist(), index= df_merc.index)

df_merc.drop(columns = ['category_name', 'name',
    'item_description','first_cat','second_cat','third_cat','brand_name', 'cat_desc'], inplace = True)

# Regression
estimators = []
estimators.append(RandomForestRegressor(n_estimators=20, n_jobs=-1,verbose=1))
#estimators.append(ExtraTreesRegressor(n_estimators=20, n_jobs=-1,verbose=1))
#estimators.append(AdaBoostRegressor())
#estimators.append(BaggingRegressor(n_estimators=10,n_jobs=-1,verbose=True))
#estimators.append(GradientBoostingRegressor(n_estimators=20, verbose=1))
#estimators.append(Ridge())
#estimators.append(SGDRegressor())
#estimators.append(Lasso())
#estimators.append(ElasticNet())
#estimators.append(SVR(verbose=True))
#estimators.append(NuSVR(verbose=True))


for est in estimators:
    train, test = train_test_split(df_merc, test_size=0.2)
    print("estimator: ", est.__class__.__name__)
    print("params: ", est.get_params())
    est.fit(train.drop(columns = 'price'), train.price)
    test['predicted'] = est.predict(test.drop(columns = 'price'))
    test['eval'] = (test['predicted'] - test['price'])**2
    eval1 = np.sqrt(1 / len(test['eval']) * test['eval'].sum())
    print("score: ", eval1)
