#!/usr/bin/python3

# Imports
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, hstack
#import matplotlib.pyplot as plt
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
#from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet
#from sklearn.svm import SVR
#from sklearn.svm import NuSVR
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
#from collections import Counter
#import re
from gensim import corpora, models, matutils
from random import sample

# Constants
MAX_DESC_WORDS = 100000 # Completly random...
MAX_NAME_WORDS = 20000 # # Completly random...
MAX_BRAND_WORDS = 5000 # # Completly random...
MAX_CAT_WORDS = 5000 # # Completly random...

# Functions

## weighing function
#def weighing(n, k, f):
#    return (1 / (1 + np.exp(- (n - k) / f)))
#
## Handling High Cardinality Categorical Features
#def ev_cat(n, ev_loc, ev_glob):
#    k = 100
#    f = 100
#    lamb = weighing(n, k, f)
#    return (lamb * ev_loc + (1 - lamb) * ev_glob)

def na_remove(df):
    df['category_name'].fillna(value = 'missing_cat', inplace = True) # To improve
    df['brand_name'].fillna(value = 'missing_brand', inplace = True) # To improve
    df['item_description'].fillna(value = 'missing_desc', inplace = True) # To improve

def csc_from_col(col, regex, max_voc): # Only Monograms for now
    stop = set(stopwords.words('english'))
    more_stop = ['i\'d', 'i\'m', 'i\'ll', ';)', '***', '**', ':)', '(:', '(;',
            ':-)', '//']

    for i in more_stop:
        stop.add(i)

    p_stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(regex)
    raw_tokens =[tokenizer.tokenize(i.lower()) for i in list(whole[col])]
    stopped_tokens = [[i for i in token if i not in stop] for token in raw_tokens]
    stem_tokens = [[p_stemmer.stem(i) for i in token] for token in stopped_tokens]
    dictionary = corpora.Dictionary(stem_tokens)
    dictionary.filter_extremes(keep_n = max_voc)
    corpus = [dictionary.doc2bow(text) for text in stem_tokens]
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]
    return matutils.corpus2csc(tfidf_corpus).transpose()

# Read and clean
print("Read and PrePreprocessing")
df_merc = pd.read_table('./train.tsv', index_col = 0)
df_merc['price'] = np.log(df_merc['price']+1)
na_remove(df_merc)
train, test = train_test_split(df_merc, test_size=0.1)
#train.drop(train[train.price == 0].index, inplace = True)
split_index = len(train)
whole = pd.concat([train, test])
print("Finished")

#df_cats = df_merc.category_name.str.split('/')
#cats = ['first_cat', 'second_cat', 'third_cat']
#
#for i in range(3):
#    df_merc[cats[i]] = df_cats.str.get(i)

# Hard clean
#df_merc.dropna(axis=0, how='any', inplace=True)


# Preprocessing categorical columns
print("Begin Preprocessing")

## Item Description
print("Begin description preprocessing")
csc_desc = csc_from_col('item_description', r'\w+', MAX_DESC_WORDS)
print("End description preprocessing")

## Name
print("Begin name preprocessing")
csc_name = csc_from_col('name', r'\w+', MAX_NAME_WORDS)
print("End name preprocessing")

## Brand Name
print("Begin name preprocessing")
csc_brand = csc_from_col('brand_name', r'\w+', MAX_BRAND_WORDS)
print("End name preprocessing")

## Category Name
print("Begin category preprocessing")
csc_cat = csc_from_col('category_name', r'(?:[^/]|//)+', MAX_CAT_WORDS)
print("End category preprocessing")

## Shipping and Item Condition
print("Begin shipping and condition preprocessing")
csc_ship_cond = csc_matrix(pd.get_dummies(whole[['shipping', 'item_condition_id']], sparse = True))
print("End shipping and condition preprocessing")

# Final csc
csc_final = hstack((csc_name, csc_brand, csc_cat, csc_ship_cond, csc_desc))
print('End Preprocessing')

#ev_price = df_merc.price.mean()
#df_merc['cat1_s']=df_merc.groupby('first_cat').price.transform('size')
#df_merc['cat1_m']=df_merc.groupby('first_cat').price.transform('mean')
#df_merc['cat1_S']=ev_cat(df_merc['cat1_s'],df_merc['cat1_m'],ev_price)
#df_merc.drop(columns = ['cat1_s','cat1_m'], inplace = True)
#
#df_merc['cat12_s']=df_merc.groupby(['first_cat','second_cat']).price.transform('size')
#df_merc['cat12_m']=df_merc.groupby(['first_cat','second_cat']).price.transform('mean')
#df_merc['cat12_S']=ev_cat(df_merc['cat12_s'],df_merc['cat12_m'],df_merc['cat1_S'])
#df_merc.drop(columns = ['cat12_s','cat12_m'], inplace = True)
#
#df_merc['cat123_s']=df_merc.groupby(['first_cat','second_cat','third_cat']).price.transform('size')
#df_merc['cat123_m']=df_merc.groupby(['first_cat','second_cat','third_cat']).price.transform('mean')
#df_merc['cat123_S']=ev_cat(df_merc['cat123_s'],df_merc['cat123_m'],df_merc['cat12_S'])
#df_merc.drop(columns = ['cat123_s','cat123_m'], inplace = True)
#
#df_merc['cat1234_s']=df_merc.groupby(['first_cat','second_cat','third_cat','brand_name']).price.transform('size')
#df_merc['cat1234_m']=df_merc.groupby(['first_cat','second_cat','third_cat','brand_name']).price.transform('mean')
#df_merc['cat1234_S']=ev_cat(df_merc['cat1234_s'],df_merc['cat1234_m'],df_merc['cat123_S'])
#df_merc.drop(columns = ['cat1234_s','cat1234_m'], inplace = True)
#df_merc.drop(columns = ['cat1_S','cat12_S','cat123_S'], inplace = True)


# Regression
estimators = []
csc_train = csc_final.tocsr()[:split_index]
csc_test = csc_final.tocsr()[split_index:]
#estimators.append(RandomForestRegressor(n_estimators=20, n_jobs=-1,verbose=1))
#estimators.append(ExtraTreesRegressor(n_estimators=20, n_jobs=-1,verbose=1))
#estimators.append(AdaBoostRegressor())
#estimators.append(BaggingRegressor(n_estimators=10,n_jobs=-1,verbose=True))
#estimators.append(GradientBoostingRegressor(n_estimators=20, verbose=1))
#estimators.append(Ridge(solver="sag", fit_intercept=True, random_state=145, alpha = 3))
#estimators.append(SGDRegressor())
#estimators.append(Lasso())
#estimators.append(ElasticNet())
#estimators.append(SVR(verbose=True))
#estimators.append(NuSVR(verbose=True))


for est in estimators:
    print("estimator: ", est.__class__.__name__)
    print("params: ", est.get_params())
    est.fit(csc_train, train.price)
    test['predicted'] = est.predict(csc_test)
    test['eval'] = (test['predicted'] - test['price'])**2
    eval1 = np.sqrt(1 / len(test['eval']) * test['eval'].sum())
    print("score: ", eval1)
