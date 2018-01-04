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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, StandardScaler

## nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

## multiprocessing
from multiprocessing import cpu_count, Pool


################
## Parameters ##
################

## run sur train split ou sur test pour submit
SUB = False

## Strat
STEMMING = True
HAS_BRAND = True
HAS_DESC = True
HAS_CAT = True
HAS_BRAND_NOW = True
IS_BUNDLE = True
NLP_DESC = True
NLP_NAME = True
NLP_BRAND = True
NLP_CAT_SPLIT = True
NLP_CAT_UNSPLIT = True
HOT_BRAND = True
NAME_LEN = True
DESC_LEN = True
DESC_LEN_POLY = True
MULTILAB_CAT = True
SHIPPING = True
CONDITION = True


## Random state
RAND = 145               # 'None', int...

## Multiproc number
N_CORES = 3

## SIGKDD
LAMBDA_K = 10            # tune EV weighted function
LAMBDA_F = 1             #          //
FUNC = 'mean'            # function to use in high-level categorical features processing

## NLP_STUFF
TOKEN_PAT = "(?u)\\b\w\w+\\b"
STEMMER = SnowballStemmer('english', ignore_stopwords = False)
STOP_W = set(stopwords.words('english'))
more_stop = set(['i\'d', 'i\'m', 'i\'ll', ';)', '***', '**', ':)', '(:', '(;', ':-)', '//'])
STOP_W |= more_stop

	
###############
## Functions ##
###############

## -------------------- Filling NaNs --------------------
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

### Apply str_extract for col of df
def feature_extract(df, col, str_list):
    return df[col].map(lambda x : str_extract(x, str_list))

### Fill NaN of col_to_fill of df with extract from col
def fill_from_extr(df, col_to_fill, col_extr, str_list):
    from_col = parallelize(feature_extract, df[col_extr][df[col_to_fill].isnull()].to_frame(), col_extr, str_list)
    df[col_to_fill].fillna(from_col, inplace=True)


## -------------------- Parallelization --------------------
### parallelize
def parallelize(func, data, col, arg):
    data_split = np.array_split(data, N_CORES) 
    pool = Pool(N_CORES)
    data = [pool.apply_async(func, args = (df, col, arg)) for df in data_split]
    data = pd.concat([d.get() for d in data])
    pool.close()
    pool.join()
    return data

## -------------------- SIGKDD --------------------
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
 
## -------------------- NLP --------------------
### stem 1 doc
def stem_word(doc,stemmer):
    tokens = [w for w in CountVectorizer(token_pattern=TOKEN_PAT).build_analyzer()(doc)]
    return " ".join([stemmer.stem(w) for w in tokens])

### stem col of df
def stem(df, col, stemmer):
    return df[col].apply(lambda doc: stem_word(doc,stemmer))

### Tf-idf col of df
def csc_from_col(df, col, token_pattern=TOKEN_PAT, min_ngram=1, max_ngram=3, max_df=0.95, min_df=2, max_features=None, idf_log=False, smooth_idf=True):
    print("Begin "+col+" NLP : " + str(datetime.datetime.now().time()))
    df_col = df[col].to_frame()
    
    if STEMMING:
    df_col = parallelize(stem, df_col, col, STEMMER)

    my_vectorizer = TfidfVectorizer(strip_accents = 'unicode',
                                    token_pattern = token_pattern,
                                    stop_words = STOP_W,
                                    ngram_range =(min_ngram, max_ngram),
                                    max_df = max_df,
                                    min_df = min_df,
                                    max_features = max_features,
                                    smooth_idf = smooth_idf,
                                    sublinear_tf = idf_log)
    my_doc_term = my_vectorizer.fit_transform(df_col)
    print(my_doc_term.shape)
    print("End "+col+" NLP : " + str(datetime.datetime.now().time()))
    return my_doc_term

## -------------------- New features --------------------
### Multilabel col of df
def multilabel(df, col, char_split = None):
    mlb = MultiLabelBinarizer(sparse_output = True)
    if char_split:
        df[col] = df[col].str.split(char_split)
    my_matrix = mlb.fit_transform(df[col])
    return my_matrix

### One hot col of df
def one_hot(df, col):
    return OneHotEncoder().fit_transform(df[col].astype('category').cat.codes.to_frame())

### Split column 'col' using delimiter 'delim'
def split_col(df, col, delim = None):
    new_cols = df[col].str.split(delim, expand = True)
    new_cols_names = [col+'_'+str(i) for i in range(new_cols.shape[1])]
    dictionary = dict(zip(new_cols.columns, new_cols_names))
    new_cols.rename(columns = dictionary, inplace = True)
    df[new_cols_names] = new_cols
    return new_cols_names

### Add feature : doc in col is Nan or not
def has_feature(df, col, name):
    df[name] = pd.notnull(df[col]).apply(int)
    return csc_matrix(df[name]).transpose()

### Change value in col of df to NaN if == string
def str_to_nan(df, col, string):
    df[col].apply(lambda x : np.nan if x == string else x)

### Count words in 1 doc
def word_count(doc, pat):
    return len([w for w in CountVectorizer(token_pattern=pat).build_analyzer()(doc)])

### Count words in col of df
def get_len(df, col, pat):
    return df[col].apply(lambda doc: word_count(doc, pat))

### Add feature word count for col of df
def add_len_feature(df, col, pat=TOKEN_PAT, p=1):
    new_col = np.power(parallelize(get_len, df, col, pat),p)
    return csc_matrix(RobustScaler().fit_transform(new_col.to_frame()))


##########
## Main ##
##########

## -------------------- Read and clean --------------------
print("Read and PrePreprocessing : " + str(datetime.datetime.now().time()))

### Read train
df_train = pd.read_table('../input/train.tsv', index_col = 0)            
sup_index_train = len(df_train)
df_train = df_train[df_train['price'] != 0]                    # drop price == 0$
df_train['price'] = np.log(df_train['price']+1)                # Price -> Log

### Read test
if SUB:
  df_test = pd.read_table('../input/test.tsv', index_col = 0)
  df_test.index = df_test.index + sup_index_train 
  
else:
  df_train, df_test = train_test_split(df_train, test_size=0.3, random_state=RAND)

split_index = len(df_train) 
print("Finished PrePreprocessing : " + str(datetime.datetime.now().time()))


## -------------------- Preprocessing --------------------
print("Begin Preprocessing : " + str(datetime.datetime.now().time()))
whole = pd.concat([df_train, df_test])

### Standardize text and Nans
print("Begin standardization : " + str(datetime.datetime.now().time()))
str_to_nan(whole, 'item_description', 'No description yet')
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
print("End feature creation : " + str(datetime.datetime.now().time()))

### Fill NaNs 1
fill_na_fast(whole, ['item_description','name'])
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
fill_na_fast(whole, 'brand_name')

### NLP
print("Begin NLP Stuff : " + str(datetime.datetime.now().time()))
if NLP_DESC:
    csc_m_list.append(csc_from_col(whole, 'item_description'))
if NLP_NAME:
    csc_m_list.append(csc_from_col(whole, 'name'))
if NLP_BRAND:
    csc_m_list.append(csc_from_col(whole, 'brand_name'))
if HOT_BRAND:
    csc_m_list.append(one_hot(whole, 'brand_name'))
if NLP_CAT_SPLIT:
    for cat in cats_list:
        csc_m_list.append(csc_from_col(whole, cat))
if NLP_CAT_UNSPLIT:
    csc_m_list.append(csc_from_col(whole, 'category_name'))
print("End NLP Stuff : " + str(datetime.datetime.now().time()))

### Length Features
print("Begin word counts : " + str(datetime.datetime.now().time()))
if NAME_LEN:
    csc_m_list.append(add_len_feature(whole, 'name'))
if DESC_LEN:
    csc_m_list.append(add_len_feature(whole, 'item_description'))
if DESC_LEN_POLY:
    csc_m_list.append(add_len_feature(whole, 'item_description', p=2))
print("End word counts : " + str(datetime.datetime.now().time()))

### Dummies
print("Begin shipping, category and condition processing : " + str(datetime.datetime.now().time()))
if IS_BUNDLE:
    whole['is_bundle'] = whole['item_description'].str.contains("\\b(bundle|joblot|lot)\\b", case = False) | whole['name'].str.contains("\\b(bundle|joblot|lot)\\b", case = False)
    csc_m_list.append(csc_matrix(whole['is_bundle']).transpose())
if SHIPPING:
    csc_m_list.append(csc_matrix(whole['shipping']).transpose())
if CONDITION:
    csc_m_list.append(csc_matrix(pd.get_dummies(whole['item_condition_id'], sparse = True)))
if MULTILAB_CAT:
    csc_m_list.append(multilabel(whole, 'category_name', '/'))
print("End shipping, category and condition processing : " + str(datetime.datetime.now().time()))

## -------------------- Final csc --------------------
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
estimator = Ridge(copy_X = False, solver="sag", fit_intercept=True, random_state=RAND, alpha = 1.5)
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
    print('End : ' + str(datetime.datetime.now().time()))
else:
    df_sub = pd.DataFrame({'test_id' : df_test.index - sup_index_train})
    df_sub['price'] = np.exp(estimator.predict(csc_test))-1
    df_sub['price'][df_sub['price'] < 3] = 3
    df_sub.to_csv('submission.csv',index=False)
