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
#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
#from collections import Counter
#import re
from gensim import corpora, models, matutils
from random import sample

# run sur train split ou sur test pour submit
SUB = False

# Strat
NLP_STRAT = 'all_nlp'     # 'no_nlp' 'mix_nlp' 'all_nlp' 'nlp+'

# Constants
#MAX_DESC_WORDS = 100000  # Completly random...
#MAX_NAME_WORDS = 20000   # Completly random...
#MAX_BRAND_WORDS = 5000   # Completly random...
#MAX_CAT_WORDS = 5000     # Completly random...
#LAMBDA_K = 10            # tune EV weighted function
#LAMBDA_F = 1             #          //
#FUNC = 'mean'          # function to use in high-level categorical features processing
stop = set(stopwords.words('english'))
more_stop = ['i\'d', 'i\'m', 'i\'ll', ';)', '***', '**', ':)', '(:', '(;',
        ':-)', '//']

for i in more_stop:
    stop.add(i)

###############
## Functions ##
###############

## fast fill nans. columns must be a list
def fill_na_fast(df, columns):
    for col in columns:
        df[col].fillna(value = 'missing_'+col, inplace = True)

## fill nans with unique values. columns must be a list
def fill_na_unique(df, columns):
    n_col = len(columns)
    n_nan = [df[columns[i]][pd.isnull(df[columns[i]])].size for i in range(n_col)]
    fill_with = [[columns[i]+'_'+str(k) for k in range(n_nan[i])] for i in range(n_col)]
    for i in range(n_col):
        df[columns[i]][pd.isnull(df[columns[i]])] = fill_with[i]
    
## weighted EV
def ev_cat(n, ev_loc, ev_glob):
    k = LAMBDA_K
    f = LAMBDA_F
    lamb = (1 / (1 + np.exp(- (n - k) / f)))
    return (lamb * ev_loc + (1 - lamb) * ev_glob)

## Compute 'new' : EV from Categorical column(s) 'cats', weighted by 'prev'
def cat_to_EV(df, cats, prev, new):
    df['tmp_size']=df.groupby(cats)['price'].transform('size')
    df['tmp_func']=df.groupby(cats)['price'].transform(FUNC)
    if type(prev) is str:
        df[new]=ev_cat(df['tmp_size'],df['tmp_func'],df[prev])
    else :
        df[new]=ev_cat(df['tmp_size'],df['tmp_func'],prev)
    df.drop(['tmp_size','tmp_func'], axis=1, inplace = True)

## EV column from 'cats' to train and merge it with test. cats must be a list
def merge_EV(train, test, cats, keep_all=False):
    ev_price_tot = train['price'].agg(FUNC)
    n_cats = len(cats)
    new = cats[0]+'_EV'
    all_cats = [new]
    if n_cats == 1:
        cat_to_EV(train,cats,ev_price_tot,new)
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
            cat_to_EV(train,cats[:(k+1)],prev,new)
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

#def build_corpus(i,regex):
#    #stem_mono_tokens = Parallel(n_jobs=4)(delayed(p_stemmer.stem)(j) for j in i if j not in stop)
#    p_stemmer = PorterStemmer()
#    tokenizer = RegexpTokenizer(regex)
#    stem_mono_tokens = [p_stemmer.stem(j) for j in tokenizer.tokenize(i.lower()) if j not in stop]
#    stem_tokens = [' '.join(i) for i in zip(stem_mono_tokens, stem_mono_tokens[1:])] + stem_mono_tokens
#    return corpora.hashdictionary.HashDictionary(stem_tokens, id_range = 20000000).doc2bow(stem_tokens)

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
#def csc_from_col(df, col, regex, max_voc, tfidf = False): # Only Monograms for now
#    #chunk_num = 3
#    #p_stemmer = PorterStemmer()
#    #tokenizer = RegexpTokenizer(regex)
#    #print("gettokens start : " + str(datetime.datetime.now().time()))
#    #stem_tokens = Parallel(n_jobs=3)(delayed(get_tokens)(i, regex) for i in list(df[col]))
#    #print("gettokens end start dictionary: " + str(datetime.datetime.now().time()))
#    #stem_tokens = [get_tokens(tokenizer.tokenize(i.lower())) for i in list(df[col])]
#    #for i in list(df[col]):
#    #    stem_mono_tokens = [p_stemmer.stem(j) for j in tokenizer.tokenize(i.lower()) if j not in stop]
#    #    stem_tokens.append([' '.join(i) for i in zip(stem_mono_tokens, stem_mono_tokens[1:])] + stem_mono_tokens)
#    #raw_tokens =tokenizer.tokenize(i.lower()) for i in list(df[col])]
#    #stopped_tokens = [[i for i in token if i not in stop] for token in raw_tokens]
#    #stem_mono_tokens = [[p_stemmer.stem(i) for i in token] for token in stopped_tokens]
#    #stem_tokens = [[' '.join(i) for i in zip(tokens, tokens[1:])] + tokens for tokens in stem_mono_tokens]
#    #gc.collect()
#    #tok_chunk = [stem_tokens[i::chunk_num] for i in range(chunk_num)]
#    #dictionary = corpora.hashdictionary.HashDictionary(stem_tokens)
#    #dictionaries = Parallel(n_jobs=3)(delayed(corpora.Dictionary)(tok) for tok in tok_chunk)
#    #print("dictionary end start dictionary_merge: " + str(datetime.datetime.now().time()))
#    #dictionary = my_dic_merge(dictionaries)
#    #print("dictionary_merge end start dictionary_filter: " + str(datetime.datetime.now().time()))
#    #dictionary.filter_extremes(keep_n = max_voc)
#    print("dictionary_filter end start corpus: " + str(datetime.datetime.now().time()))
#    #corpus = [hashdictionary.HashDictionary(text).doc2bow(text) for text in stem_tokens]
#    corpus = Parallel(n_jobs = 3)(delayed(build_corpus)(i, regex) for i in list(df[col]))
#    if tfidf:
#        tfidf = models.TfidfModel(corpus)
#        tfidf_corpus = tfidf[corpus]
#        print("corpus end start return: " + str(datetime.datetime.now().time()))
#        my_csc_matrix = matutils.corpus2csc(tfidf_corpus).transpose()
#    else:
#        print("corpus end start return: " + str(datetime.datetime.now().time()))
#        my_csc_matrix = matutils.corpus2csc(corpus).transpose()
#    return my_csc_matrix

# corpus2csc
def csc_from_col(df, col, tfidf = True, min_ngram = 1, max_ngram = 3, max_df = 1.0, min_df = 1, max_features = 2000000, idf_log = False, smooth_idf = True):
#def csc_from_col(df, col, tfidf = True, min_ngram = 1, max_ngram = 1, max_df = 1.0, min_df = 1, max_features = 20000000, idf_log = False, smooth_idf = True):

    #def stemming(doc):
    #    return (my_stemmer.stem(w) for w in my_analyzer(doc))

    #my_stemmer = SnowballStemmer('english', ignore_stopwords = True)
    #my_analyzer = CountVectorizer().build_analyzer()
    my_vectorizer = CountVectorizer(stop_words = stop,
                                    ngram_range = (min_ngram, max_ngram), max_df = max_df,
                                    min_df = min_df,
                                    max_features = max_features)
                                    #analyzer = stemming)
    my_doc_term = my_vectorizer.fit_transform(df[col])
    print(my_doc_term.shape)
    if tfidf:
        tfidf_trans = TfidfTransformer(smooth_idf = smooth_idf, sublinear_tf = idf_log)
        my_doc_term = tfidf_trans.fit_transform(my_doc_term)
    return my_doc_term

def one_hot(df, col):
    enc = OneHotEncoder()
    my_matrix = enc.fit_transform(df[col].astype('category').cat.codes.to_frame())
    print(my_matrix.shape)
    return my_matrix

def multilabel(df, col, char_split = None):
    mlb = MultiLabelBinarizer(sparse_output = True)
    if char_split:
        df[col] = df[col].str.split(char_split)
    my_matrix = mlb.fit_transform(df[col])
    print(my_matrix.shape)
    return my_matrix

##########
## Main ##
##########

# Read and clean
print("Read and PrePreprocessing : " + str(datetime.datetime.now().time()))

## Read train
df_train = pd.read_table('../input/train.tsv', index_col = 0)             # for kaggle kernel
df_train['price'] = np.log(df_train['price']+1)  # Price -> Log
df_train = df_train[df_train['price'] != 0]      # drop price == 0$

## Read test
#df_test = pd.read_table('../input/test.tsv', index_col = 0)

## fill nans
fill_na_fast(df_train, ['item_description','name'])
fill_na_unique(df_train, ['category_name','brand_name'])

## if run only on train
if not SUB:
    df_train, df_test = train_test_split(df_train, test_size=0.3)

split_index = len(df_train) 
print("Finished PrePreprocessing : " + str(datetime.datetime.now().time()))
####################################

# Preprocessing categorical columns
print("Begin Preprocessing : " + str(datetime.datetime.now().time()))
whole = pd.concat([df_train, df_test])
csc_desc, csc_name, csc_brand, csc_brand_SI, csc_cat, csc_cat_SI, csc_ship_cond = None, None, None, None, None, None, None

print("Begin description preprocessing : " + str(datetime.datetime.now().time()))
if NLP_STRAT != 'no_nlp':
    #csc_desc = csc_from_col(whole, 'item_description', r'\w+', MAX_DESC_WORDS, tfidf = True)
    csc_desc = csc_from_col(whole, 'item_description')
print("End description preprocessing : " + str(datetime.datetime.now().time()))
    
print("Begin name preprocessing : " + str(datetime.datetime.now().time()))
if NLP_STRAT != 'no_nlp':
    #csc_name = csc_from_col(whole, 'name', r'\w+', MAX_NAME_WORDS)
    csc_name = csc_from_col(whole, 'name')
print("End name preprocessing : " + str(datetime.datetime.now().time()))
    
print("Begin brand name preprocessing : " + str(datetime.datetime.now().time()))
if (NLP_STRAT == 'all_nlp') or (NLP_STRAT == 'nlp+'):
    #csc_brand = csc_from_col(whole, 'brand_name', r'\w+', MAX_BRAND_WORDS)
    #csc_brand = csc_from_col(whole, 'brand_name')
    csc_brand = one_hot(whole, 'brand_name')
if NLP_STRAT != 'all_nlp':
    train_brand, test_brand = merge_EV(df_train, df_test, ['brand_name'])
    csc_brand_SI = csc_matrix(pd.concat([train_brand, test_brand]))
print("End brand name preprocessing : " + str(datetime.datetime.now().time()))

print("Begin category preprocessing : " + str(datetime.datetime.now().time()))
if (NLP_STRAT == 'all_nlp') or (NLP_STRAT == 'nlp+'):
    #csc_cat = csc_from_col(whole, 'category_name', r'(?:[^/]|//)+', MAX_CAT_WORDS)
    #csc_cat = csc_from_col(whole, 'category_name')
    csc_cat = multilabel(whole, 'category_name', '/')
if NLP_STRAT != 'all_nlp':
    train_cat, test_cat = merge_EV(df_train, df_test, ['category_name'])
    csc_cat_SI = csc_matrix(pd.concat([train_cat, test_cat]))
print("End category preprocessing : " + str(datetime.datetime.now().time()))

print("Begin shipping and condition preprocessing : " + str(datetime.datetime.now().time()))
csc_ship_cond = csc_matrix(pd.get_dummies(whole[['shipping', 'item_condition_id']], sparse = True))
#csc_ship_cond = whole[['shipping','item_condition_id']]
print("End shipping and condition preprocessing : " + str(datetime.datetime.now().time()))

## Final csc
csc_final = hstack((csc_desc, csc_name, csc_brand, csc_brand_SI, csc_cat, csc_cat_SI, csc_ship_cond))
print("csc_final shape : " + str(csc_final.shape))
#print("csc_final non zero : " + str(csc_final.count_nonzero()))
#print("csc_final sparsity : " + str(csc_final.count_nonzero()/(csc_final.shape[0]*csc_final.shape[1])))
csc_train = csc_final.tocsr()[:split_index]
csc_test = csc_final.tocsr()[split_index:]
print("End Preprocessing : " + str(datetime.datetime.now().time()))
######################################


### Regression ###
estimator = None
#estimator = RandomForestRegressor(n_estimators=20, n_jobs=-1,verbose=1)
#estimator = ExtraTreesRegressor(n_estimators=20, n_jobs=-1,verbose=1)
#estimator = AdaBoostRegressor()
#estimator = BaggingRegressor(n_estimators=10,n_jobs=-1,verbose=True)
#estimator = GradientBoostingRegressor(n_estimators=20, verbose=1)
estimator = Ridge(solver="sag", fit_intercept=True, random_state=145, alpha = 2)
#estimator = SGDRegressor()
#estimator = Lasso()
#estimator = ElasticNet()
#estimator = SVR(verbose=True)
#estimator = NuSVR(verbose=True)
print("estimator: ", estimator.__class__.__name__)
print("params: ", estimator.get_params())
estimator.fit(csc_train, df_train.price)

if not SUB:
    df_test['predicted'] = estimator.predict(csc_test)
    df_test['eval'] = (df_test['predicted'] - df_test['price'])**2
    eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
    print("score: ", eval1)
else:
    df_sub = pd.DataFrame({'test_id':df_test.index})
    df_sub['price'] = np.exp(estimator.predict(csc_test))-1
    df_sub.to_csv('submission.csv',index=False)
