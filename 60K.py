#!/usr/bin/python3

# Imports
## del warnings
import warnings
import datetime
import gc
warnings.filterwarnings("ignore")
## basics
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, hstack
## sklearn
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet
#from sklearn.svm import SVR
#from sklearn.svm import NuSVR
## nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from joblib import Parallel, delayed
#from collections import Counter
#import re
from gensim import corpora, models, matutils
from random import sample



# Constants
MAX_DESC_WORDS = 100000  # Completly random...
MAX_NAME_WORDS = 20000   # Completly random...
MAX_BRAND_WORDS = 5000   # Completly random...
MAX_CAT_WORDS = 5000     # Completly random...
LAMBDA_K = 10            # tune EV weighted function
LAMBDA_F = 1             #          //
FUNC = 'mean'          # function to use in high-level categorical features processing
stop = set(stopwords.words('english'))
more_stop = ['i\'d', 'i\'m', 'i\'ll', ';)', '***', '**', ':)', '(:', '(;',
        ':-)', '//']

for i in more_stop:
    stop.add(i)



###############
## Functions ##
###############

## fast fill nans
def na_remove(df):
    df['category_name'].fillna(value = 'missing_cat', inplace = True) # To improve
    df['brand_name'].fillna(value = 'missing_brand', inplace = True) # To improve
    df['item_description'].fillna(value = 'missing_desc', inplace = True) # To improve
    df['name'].fillna(value = 'missing_name', inplace = True) # To improve
    
## weighted EV
def ev_cat(n, ev_loc, ev_glob):
    k = LAMBDA_K
    f = LAMBDA_F
    lamb = (1 / (1 + np.exp(- (n - k) / f)))
    return (lamb * ev_loc + (1 - lamb) * ev_glob)

## Compute 'new' : EV from Categorical column(s) 'cats', weighted by 'prev'
def cat_to_EV(dfc, cats, prev, new):
    df = dfc
    df['tmp_size']=df.groupby(cats)['price'].transform('size')
    df['tmp_func']=df.groupby(cats)['price'].transform(FUNC)
    if type(prev) is str:
        df[new]=ev_cat(df['tmp_size'],df['tmp_func'],df[prev])
    else :
        df[new]=ev_cat(df['tmp_size'],df['tmp_func'],prev)
    df.drop(['tmp_size','tmp_func'], axis=1, inplace = True)
    return df

## EV column from 'cats' to train and merge it with test. cats must be a list
def merge_EV(train, test, cats, keep_all=False):
    ev_price_tot = train['price'].agg(FUNC)
    n_cats = len(cats)
    new = cats[0]+'_EV'
    all_cats = [new]
    if n_cats == 1:
        train = cat_to_EV(train,cats,ev_price_tot,new)
        cor_cats=train.groupby(cats)[new].first().to_frame().reset_index()
        test = pd.merge(test, cor_cats, on=cats, how='left')
        test[new][pd.isnull(test[new])]=ev_price_tot
    else:
        prev = ''
        cor_cats=[]
        for k in range(n_cats):
            if k == 0:
                prev = ev_price_tot
            else:
                prev = new
            new += str(k)
            all_cats.append(new)
            train = cat_to_EV(train,cats[:(k+1)],prev,new)
            cor_cats.append(train.groupby(cats[:(k+1)])[new].first().to_frame().reset_index())
            test = pd.merge(test, cor_cats[k], on=cats[:(k+1)], how='left')
            if k == 0:
                test[new][pd.isnull(test[new])]=ev_price_tot
            else:
                test[new][pd.isnull(test[new])]=test[prev]    
    if not keep_all:
        return train[all_cats[(len(all_cats)-1):]], test[all_cats[(len(all_cats)-1):]]
    else:
        return train[all_cats], test[all_cats]
 
# For parallel
#def get_tokens(i,regex):
#    #stem_mono_tokens = Parallel(n_jobs=4)(delayed(p_stemmer.stem)(j) for j in i if j not in stop)
#    p_stemmer = PorterStemmer()
#    tokenizer = RegexpTokenizer(regex)
#    stem_mono_tokens = [p_stemmer.stem(j) for j in tokenizer.tokenize(i.lower()) if j not in stop]
#    return [' '.join(i) for i in zip(stem_mono_tokens, stem_mono_tokens[1:])] + stem_mono_tokens

def build_corpus(i,regex):
    #stem_mono_tokens = Parallel(n_jobs=4)(delayed(p_stemmer.stem)(j) for j in i if j not in stop)
    p_stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(regex)
    stem_mono_tokens = [p_stemmer.stem(j) for j in tokenizer.tokenize(i.lower()) if j not in stop]
    stem_tokens = [' '.join(i) for i in zip(stem_mono_tokens, stem_mono_tokens[1:])] + stem_mono_tokens
    return corpora.hashdictionary.HashDictionary(stem_tokens, id_range = 20000000).doc2bow(stem_tokens)

#def my_dic_merge(dic_list):
#    dic_len = len(dic_list[0])
#    a = dic_list[0]
#    for b in dic_list[1:]:
#        for k in b.token2id:
#            if k not in set(a.token2id):
#                a.token2id[k] = dic_len
#                dic_len += 1
#
#    return a

## corpus2csc
def csc_from_col(df, col, regex, max_voc, tfidf = False): # Only Monograms for now
    #chunk_num = 3
    #p_stemmer = PorterStemmer()
    #tokenizer = RegexpTokenizer(regex)
    #print("gettokens start : " + str(datetime.datetime.now().time()))
    #stem_tokens = Parallel(n_jobs=3)(delayed(get_tokens)(i, regex) for i in list(df[col]))
    #print("gettokens end start dictionary: " + str(datetime.datetime.now().time()))
    #stem_tokens = [get_tokens(tokenizer.tokenize(i.lower())) for i in list(df[col])]
    #for i in list(df[col]):
    #    stem_mono_tokens = [p_stemmer.stem(j) for j in tokenizer.tokenize(i.lower()) if j not in stop]
    #    stem_tokens.append([' '.join(i) for i in zip(stem_mono_tokens, stem_mono_tokens[1:])] + stem_mono_tokens)
    #raw_tokens =tokenizer.tokenize(i.lower()) for i in list(df[col])]
    #stopped_tokens = [[i for i in token if i not in stop] for token in raw_tokens]
    #stem_mono_tokens = [[p_stemmer.stem(i) for i in token] for token in stopped_tokens]
    #stem_tokens = [[' '.join(i) for i in zip(tokens, tokens[1:])] + tokens for tokens in stem_mono_tokens]
    #gc.collect()
    #tok_chunk = [stem_tokens[i::chunk_num] for i in range(chunk_num)]
    #dictionary = corpora.hashdictionary.HashDictionary(stem_tokens)
    #dictionaries = Parallel(n_jobs=3)(delayed(corpora.Dictionary)(tok) for tok in tok_chunk)
    #print("dictionary end start dictionary_merge: " + str(datetime.datetime.now().time()))
    #dictionary = my_dic_merge(dictionaries)
    #print("dictionary_merge end start dictionary_filter: " + str(datetime.datetime.now().time()))
    #dictionary.filter_extremes(keep_n = max_voc)
    print("dictionary_filter end start corpus: " + str(datetime.datetime.now().time()))
    #corpus = [hashdictionary.HashDictionary(text).doc2bow(text) for text in stem_tokens]
    corpus = Parallel(n_jobs = 3)(delayed(build_corpus)(i, regex) for i in list(df[col]))
    if tfidf:
        tfidf = models.TfidfModel(corpus)
        tfidf_corpus = tfidf[corpus]
        print("corpus end start return: " + str(datetime.datetime.now().time()))
        my_csc_matrix = matutils.corpus2csc(tfidf_corpus).transpose()
    else:
        print("corpus end start return: " + str(datetime.datetime.now().time()))
        my_csc_matrix = matutils.corpus2csc(corpus).transpose()
    return my_csc_matrix[:, my_csc_matrix.getnnz(0) > 400]
    #return matutils.corpus2csc(tfidf_corpus).transpose()


##########
## Main ##
##########

# Read and clean
print("Read and PrePreprocessing : " + str(datetime.datetime.now().time()))

## Read train
df_train = pd.read_table('./train.tsv', index_col = 0)
#df_train = pd.read_table('../input/train.tsv', index_col = 0)             # for kaggle kernel
df_train['price'] = np.log(df_train['price']+1)  # Price -> Log
df_train = df_train[df_train['price'] != 0]      # drop price == 0$

## Read test
#df_test = pd.read_table('../input/test.tsv', index_col = 0)

### if run only on train ###
df_train, df_test = train_test_split(df_train, test_size=0.3)

split_index = len(df_train)                      
whole = pd.concat([df_train, df_test])
na_remove(whole)                   # fill nans

print("Finished : " + str(datetime.datetime.now().time()))
####################################


# Preprocessing categorical columns
print("Begin Preprocessing : " + str(datetime.datetime.now().time()))

## Item Description
print("Begin description preprocessing : " + str(datetime.datetime.now().time()))
csc_desc = csc_from_col(whole, 'item_description', r'\w+', MAX_DESC_WORDS, tfidf = True)
print("End description preprocessing : " + str(datetime.datetime.now().time()))

## Name
print("Begin name preprocessing : " + str(datetime.datetime.now().time()))
csc_name = csc_from_col(whole, 'name', r'\w+', MAX_NAME_WORDS)
print("End name preprocessing : " + str(datetime.datetime.now().time()))

## Brand Name
print("Begin name preprocessing : " + str(datetime.datetime.now().time()))
csc_brand = csc_from_col(whole, 'brand_name', r'\w+', MAX_BRAND_WORDS)
#train_brand, test_brand = merge_EV(df_train, df_test, ['brand_name'])
#csc_brand = csc_matrix(pd.concat([train_brand, test_brand]))
print("End name preprocessing : " + str(datetime.datetime.now().time()))

## Category Name
print("Begin category preprocessing : " + str(datetime.datetime.now().time()))
csc_cat = csc_from_col(whole, 'category_name', r'(?:[^/]|//)+', MAX_CAT_WORDS)
#train_cat, test_cat = merge_EV(df_train, df_test, ['category_name'])
#csc_cat = csc_matrix(pd.concat([train_cat, test_cat]))
print("End category preprocessing : " + str(datetime.datetime.now().time()))

## Shipping and Item Condition
print("Begin shipping and condition preprocessing : " + str(datetime.datetime.now().time()))
csc_ship_cond = csc_matrix(pd.get_dummies(whole[['shipping', 'item_condition_id']], sparse = True))
#csc_ship_cond = whole[['shipping','item_condition_id']]
print("End shipping and condition preprocessing : " + str(datetime.datetime.now().time()))

## Final csc
csc_final = hstack((csc_name, csc_brand, csc_cat, csc_ship_cond, csc_desc))
print("csc_final shape : " + str(csc_final.shape))
print("csc_final non zero : " + str(csc_final.count_nonzero()))
print("csc_final sparsity : " + str(csc_final.count_nonzero()/(csc_final.shape[0]*csc_final.shape[1])))
print("End Preprocessing : " + str(datetime.datetime.now().time()))
######################################



################
## Regression ##
################

estimators = []
csc_train = csc_final.tocsr()[:split_index]
csc_test = csc_final.tocsr()[split_index:]
#estimators.append(RandomForestRegressor(n_estimators=20, n_jobs=-1,verbose=1))
#estimators.append(ExtraTreesRegressor(n_estimators=20, n_jobs=-1,verbose=1))
#estimators.append(AdaBoostRegressor())
#estimators.append(BaggingRegressor(n_estimators=10,n_jobs=-1,verbose=True))
#estimators.append(GradientBoostingRegressor(n_estimators=20, verbose=1))
estimators.append(Ridge(solver="sag", fit_intercept=True, random_state=145, alpha = 0.7))
#estimators.append(SGDRegressor())
#estimators.append(Lasso())
#estimators.append(ElasticNet())
#estimators.append(SVR(verbose=True))
#estimators.append(NuSVR(verbose=True))


for est in estimators:
    print("estimator: ", est.__class__.__name__)
    print("params: ", est.get_params())
    est.fit(csc_train, df_train.price)
    df_test['predicted'] = est.predict(csc_test)
    df_test['eval'] = (df_test['predicted'] - df_test['price'])**2
    eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
    print("score: ", eval1)

    '''est.fit(df_train.drop(['price'], axis=1), df_train.price)
    df_sub = pd.DataFrame({'test_id':df_test.index})
    df_sub['price'] = np.exp(est.predict(df_test))
    df_sub.to_csv('submission.csv',index=False)'''
