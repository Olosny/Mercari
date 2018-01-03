#!/usr/bin/python3

#############
## Imports ##
#############

## Obs
import datetime
import warnings
warnings.filterwarnings("ignore")

## basics
import pandas as pd
import numpy as np
import re
#import sys

## scipy
from scipy.sparse import csc_matrix, csr_matrix, hstack

## sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, SGDRegressor, HuberRegressor
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer
## nltk
from nltk.corpus import stopwords
#from nltk.stem.snowball import SnowballStemmer

## multiprocessing
from multiprocessing import cpu_count, Pool


################
## Parameters ##
################

## run sur train split ou sur test pour submit
SUB = False

## Strat
HAS_BRAND = True
HAS_DESC = True
HAS_CAT = True
HAS_BRAND_NOW = True
NLP_DESC = True
NLP_NAME = True
NLP_CAT_SPLIT = True
NLP_CAT_UNSPLIT = True
NLP_BRAND = True
NAME_LEN = True
DESC_LEN = True
DESC_LEN_POLY = True
MULTILAB_CAT = True
SHIPPING = True
CONDITION = True


## Random state
RAND = 0                 # 'None', int...

## Multiproc number
N_CORES = 3

## SIGKDD
LAMBDA_K = 10            # tune EV weighted function
LAMBDA_F = 1             #          //
FUNC = 'mean'            # function to use in high-level categorical features processing

## English stop words
stop = set(stopwords.words('english'))
more_stop = ['i\'d', 'i\'m', 'i\'ll', ';)', '***', '**', ':)', '(:', '(;', ':-)', '//']

for i in more_stop:
    stop.add(i)

	
###############
## Functions ##
###############

## Filling NaNs ---------------------------------------------------------------
### fast fill nans
def fill_na_fast(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        df[col].fillna(value = 'missing_'+col, inplace = True)

### fill nans with unique values
def fill_na_unique(df, columns):
    if isinstance(columns, str):
        columns = [columns]
    n_col = len(columns)
    n_nan = [df[columns[i]][pd.isnull(df[columns[i]])].size for i in range(n_col)]
    fill_with = [[columns[i]+'_'+str(k) for k in range(n_nan[i])] for i in range(n_col)]
    for i in range(n_col):
        df[columns[i]][pd.isnull(df[columns[i]])] = fill_with[i]

### find if an element of 'str_list' is in 'string'
def str_extract(string, str_list):
    match = re.search(str_list, string, flags=re.IGNORECASE)
    extr = match.group(0) if match else np.nan
    return extr	

### 
def feature_extract(df, col, str_list):
    return df[col].map(lambda x : str_extract(x, str_list))

###
def fill_from_extr(df, col_to_fill, col_extr, str_list):
    from_col = parallelize(feature_extract, df[col_extr][df[col_to_fill].isnull()].to_frame(), col_extr, str_list)
    df[col_to_fill].fillna(from_col, inplace=True)


## Parallelization ------------------------------------------------------------
### parallelize
def parallelize(func, data, col, arg):
    data_split = np.array_split(data, N_CORES) 
    pool = Pool(N_CORES)
    data = [pool.apply_async(func, args = (df, col, arg)) for df in data_split]
    data = pd.concat([d.get() for d in data])
    pool.close()
    pool.join()
    return data
    
## SIGKDD ---------------------------------------------------------------------
### weighted EV
def ev_cat(n, ev_loc, ev_glob):
    k = LAMBDA_K
    f = LAMBDA_F
    lamb = (1 / (1 + np.exp(- (n - k) / f)))
    return (lamb * ev_loc + (1 - lamb) * ev_glob)

### Compute 'new' : EV from Categorical column(s) 'cats', weighted by 'prev'
def cat_to_EV(df, cats, prev, new):
    df['tmp_size']=df.groupby(cats)['price'].transform('size')
    df['tmp_func']=df.groupby(cats)['price'].transform(FUNC)
    if type(prev) is str:
        df[new]=ev_cat(df['tmp_size'],df['tmp_func'],df[prev])
    else :
        df[new]=ev_cat(df['tmp_size'],df['tmp_func'],prev)
    df.drop(['tmp_size','tmp_func'], axis=1, inplace = True)

### EV column from 'cats' to train and merge it with test. cats must be a list
def merge_EV(train, test, cats, keep_all=False):
    if isinstance(cats,str):
        cats = [cats]
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
 
## NLP ------------------------------------------------------------------------
### corpus2csc
def csc_from_col(df, col, tfidf = True, min_ngram = 1, max_ngram = 3, max_df = 1.0, min_df = 1, max_features = 2000000, idf_log = False, smooth_idf = True):
#def csc_from_col(df, col, tfidf = True, min_ngram = 1, max_ngram = 1, max_df = 1.0, min_df = 1, max_features = 20000000, idf_log = False, smooth_idf = True):
    print("Begin "+col+" NLP : " + str(datetime.datetime.now().time()))

    #def stemming(doc):
    #    return (my_stemmer.stem(w) for w in my_analyzer(doc))

    #my_stemmer = SnowballStemmer('english', ignore_stopwords = True)
    #my_analyzer = CountVectorizer().build_analyzer()
    my_vectorizer = CountVectorizer(stop_words = stop,
                                    ngram_range = (min_ngram, max_ngram),
                                    max_df = max_df,
                                    min_df = min_df,
                                    max_features = max_features)
                                    #analyzer = stemming)
    my_doc_term = my_vectorizer.fit_transform(df[col])
    if tfidf:
        tfidf_trans = TfidfTransformer(smooth_idf = smooth_idf, sublinear_tf = idf_log)
        my_doc_term = tfidf_trans.fit_transform(my_doc_term)
    print("End "+col+" NLP : " + str(datetime.datetime.now().time()))
    return my_doc_term

## ??? ------------------------------------------------------------------------
###
def multilabel(df, col, char_split = None):
    mlb = MultiLabelBinarizer(sparse_output = True)
    if char_split:
        df[col] = df[col].str.split(char_split)
    my_matrix = mlb.fit_transform(df[col])
    return my_matrix

### Split column 'col' using delimiter 'delim'
def split_col(df, col, delim = None):
    new_cols = df[col].str.split(delim, expand = True)
    new_cols_names = [col+'_'+str(i) for i in range(new_cols.shape[1])]
    dictionary = dict(zip(new_cols.columns, new_cols_names))
    new_cols.rename(columns = dictionary, inplace = True)
    df[new_cols_names] = new_cols
    return new_cols_names


###
def has_feature(df, col, name):
    df[name] = pd.notnull(df[col]).apply(int)
    return csc_matrix(df[name]).transpose()

###
def no_desc_to_nan(df,col,str):
    df[col].apply(lambda x : np.nan if x == str else x)

###
def tokenize(df, columns, stop_w = None):
    if isinstance(columns, str):
        columns = [columns]
    vectorizer = CountVectorizer(stop_words = stop_w)
    new_cols = []
    for col in columns:
        new_cols.append(col+'_token')
        df[col+'_token'] = df[col].apply(vectorizer.build_analyzer())
    return new_cols

###
def word_count(df, col, name, func = lambda x: x):
    df[name] = df[col].apply(len)
    df[name] = df[name].apply(func)
    df[name] = RobustScaler().fit_transform(df[name].to_frame())
    return csc_matrix(df[name]).transpose()



##########
## Main ##
##########

## Read and clean -------------------------------------------------------------
print("Read and Pre-Preprocessing : " + str(datetime.datetime.now().time()))

### Read train
df_train = pd.read_table('../input/train.tsv', index_col = 0)            
sup_index_train = len(df_train)
df_train = df_train[df_train['price'] != 0]                    # drop price == 0$
df_train['price'] = np.log(df_train['price']+1)                # take log of price

### Read test
if SUB:
  df_test = pd.read_table('../input/test.tsv', index_col = 0)
  df_test.index = df_test.index + sup_index_train

else:	
    df_train, df_test = train_test_split(df_train, test_size=0.3, random_state=RAND)

split_index = len(df_train) 
print("Finished Pre-Preprocessing : " + str(datetime.datetime.now().time()))


## Preprocessing --------------------------------------------------------------
print("Begin Preprocessing : " + str(datetime.datetime.now().time()))
whole = pd.concat([df_train, df_test])

### Standardize text and Nans
print("Begin standardization : " + str(datetime.datetime.now().time()))
no_desc_to_nan(whole, 'item_description', 'No description yet')
whole = whole.applymap(lambda x: x if type(x)!=str else x.lower())
print("End standardization : " + str(datetime.datetime.now().time()))

csc_m_list=[]

### Feature creation
print("Begin feature creation : " + str(datetime.datetime.now().time()))
if HAS_BRAND:
    csc_m_list.append(has_feature(whole, 'brand_name', 'has_brand'))
if HAS_DESC:
    csc_m_list.append(has_feature(whole, 'item_description', 'has_item_description'))
if HAS_CAT:
    csc_m_list.append(has_feature(whole, 'category_name', 'has_category_name'))

# To do : check name
print("End feature creation : " + str(datetime.datetime.now().time()))

### Fill NaNs 1
fill_na_fast(whole, ['item_description','name'])
feature_token = tokenize(whole, ['name','item_description'])
cats_list = split_col(whole, 'category_name', delim = '/')
fill_na_fast(whole, ['category_name']+cats_list)
### Extract some missing brand names
print("Start extracting brand names : " + str(datetime.datetime.now().time()))
brands_list = sorted(whole['brand_name'][whole['brand_name'].notnull()].unique(), key = len, reverse = True)
brands_list = "\\b("+"|".join([re.escape(m) for m in brands_list])+")\\b"
fill_from_extr(whole, 'brand_name', 'name', brands_list)
fill_from_extr(whole, 'brand_name', 'item_description', brands_list)
if HAS_BRAND_NOW:
    csc_m_list.append(has_feature(whole, 'brand_name', 'has_brand_now'))
print("End extracting brand names : " + str(datetime.datetime.now().time()))

### Fill NaNs 2
fill_na_fast(whole, ['brand_name'])

### NLP
print("Begin NLP Stuff : " + str(datetime.datetime.now().time()))
if NLP_DESC:
    csc_m_list.append(csc_from_col(whole, 'item_description'))
if NLP_NAME:
    csc_m_list.append(csc_from_col(whole, 'name'))
if NLP_BRAND:
    csc_m_list.append(csc_from_col(whole, 'brand_name'))
if NLP_CAT_SPLIT:
    for cat in cats_list:
        csc_m_list.append(csc_from_col(whole, cat))
if NLP_CAT_UNSPLIT:
    csc_m_list.append(csc_from_col(whole, 'category_name'))
print("End NLP Stuff : " + str(datetime.datetime.now().time()))
  
### Length Features
if NAME_LEN:
    csc_m_list.append(word_count(whole, 'name_token', 'name_len'))
if DESC_LEN:
    csc_m_list.append(word_count(whole, 'item_description_token', 'desc_len'))
if DESC_LEN_POLY:
	csc_m_list.append(word_count(whole, 'item_description_token', 'desc_len', lambda x: x^2))
### Dummies
print("Begin shipping and condition processing : " + str(datetime.datetime.now().time()))
if SHIPPING:
    csc_m_list.append(csc_matrix(whole['shipping']).transpose())
if CONDITION:
    csc_m_list.append(csc_matrix(pd.get_dummies(whole['item_condition_id'], sparse = True)))
if MULTILAB_CAT:
    csc_m_list.append(multilabel(whole, 'category_name', '/'))
print("End shipping and condition processing : " + str(datetime.datetime.now().time()))

## Final csc ..................................................................
csc_final = hstack(csc_m_list)
print("csc_final shape : " + str(csc_final.shape))
csc_train = csc_final.tocsr()[:split_index]
csc_test = csc_final.tocsr()[split_index:]
print("End Preprocessing : " + str(datetime.datetime.now().time()))


################
## Regression ##
################

#estimator = RandomForestRegressor(n_estimators=20, n_jobs=-1,verbose=1)
#estimator = ExtraTreesRegressor(n_estimators=20, n_jobs=-1,verbose=1)
#estimator = AdaBoostRegressor()
#estimator = BaggingRegressor(n_estimators=10,n_jobs=-1,verbose=True)
#estimator = GradientBoostingRegressor(n_estimators=20, verbose=1)
estimator = Ridge(solver="sag", fit_intercept=True, random_state=RAND, alpha = 2)
#estimator = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)
#estimator = SGDRegressor(loss=’squared_loss’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, max_iter=5, tol=None, shuffle=True, epsilon=0.1, random_state=RAND, learning_rate=’invscaling’, eta0=0.01, power_t=0.25)
#estimator = Lasso()
#estimator = ElasticNet()
#estimator = SVR(verbose=True)
#estimator = NuSVR(verbose=True)
print("estimator: ", estimator.__class__.__name__)
print("params: ", estimator.get_params())
estimator.fit(csc_train, df_train.price)

if not SUB:
    df_test['predicted'] = estimator.predict(csc_test)
    df_test['predicted'] = np.exp(df_test['predicted'])-1
    df_test['predicted'][df_test['predicted'] < 3] = 3
    df_test['predicted'] = np.log(df_test['predicted']+1)
    df_test['eval'] = (df_test['predicted'] - df_test['price'])**2
    eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
    print("score: ", eval1)
else:
   df_sub = pd.DataFrame({'test_id' : df_test.index - sup_index_train})
   df_sub['price'] = np.exp(estimator.predict(csc_test))-1
   df_sub['price'][df_sub['price'] < 3] = 3
   df_sub.to_csv('submission.csv',index=False)
