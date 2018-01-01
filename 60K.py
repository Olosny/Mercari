#!/usr/bin/python3

# Imports

## del warnings
import warnings
warnings.filterwarnings("ignore")

## basics
import pandas as pd
import numpy as np
import datetime
import re
#import sys

## scipy
from scipy.sparse import csc_matrix, csr_matrix, hstack

## sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, RobustScaler

## nltk
from nltk.corpus import stopwords
#from nltk.stem.snowball import SnowballStemmer

## multiprocessing
from multiprocessing import cpu_count, Pool

# run sur train split ou sur test pour submit
SUB = False

# Strat
NLP_STRAT = 'all_nlp'     # 'no_nlp' 'mix_nlp' 'all_nlp' 'nlp+'

# Multiproc number
cores = 3

# English stop words
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
 
# corpus2csc
def csc_from_col(df, col, tfidf = True, min_ngram = 1, max_ngram = 3, max_df = 1.0, min_df = 1, max_features = 2000000, idf_log = False, smooth_idf = True):

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

def parallelize(func, data, col, arg):
    data_split = np.array_split(data, cores) 
    pool = Pool(cores)
    data = [pool.apply_async(func, args = (df, col, arg)) for df in data_split]
    data = pd.concat([d.get() for d in data])
    pool.close()
    pool.join()
    return data

def str_extract(string, str_list):
    match = re.search(str_list, string, flags=re.IGNORECASE)
    extr = match.group(0) if match else np.nan
    return extr

def brand_extract(df, col, str_list):
    return df[col].map(lambda x : str_extract(x, str_list))

def has_desc(df):
    return df['item_description'].apply(lambda x : x != 'No description yet' and x != 'missing_item_description')

##########
## Main ##
##########

# Read and clean
print("Read and PrePreprocessing : " + str(datetime.datetime.now().time()))

## Read train
df_train = pd.read_table('../input/train.tsv', index_col = 0)            
df_train = df_train[df_train['price'] != 0]                    # drop price == 0$
df_train['price'] = np.log(df_train['price']+1)                # Price -> Log

## Read test
if SUB:
  df_test = pd.read_table('../input/test.tsv', index_col = 0)
  
else:
  df_train, df_test = train_test_split(df_train, test_size=0.3)


split_index = len(df_train) 
print("Finished PrePreprocessing : " + str(datetime.datetime.now().time()))
####################################

# Preprocessing
print("Begin Preprocessing : " + str(datetime.datetime.now().time()))
whole = pd.concat([df_train, df_test])
csc_desc, csc_name, csc_brand, csc_brand_SI, csc_cat, csc_cat_SI, csc_ship, csc_cond, csc_has_brand, csc_has_cat, csc_has_desc = None, None, None, None, None, None, None, None, None, None, None

## Column creation and na fill
### ez ops
print("Begin column creation and na fill : " + str(datetime.datetime.now().time()))
fill_na_fast(whole, ['item_description','name'])
whole['has_brand'] = pd.notnull(whole['brand_name']).apply(int)
whole['has_cat'] = pd.notnull(whole['category_name']).apply(int)
whole['has_desc'] = has_desc(whole).apply(int)
csc_has_brand = csc_matrix(whole['has_brand']).transpose()
csc_has_cat = csc_matrix(whole['has_cat']).transpose()
csc_has_desc = csc_matrix(whole['has_desc']).transpose()
# To do : check name

### Brand name fill (to refactor with a function)
brands_list = sorted(whole['brand_name'][whole['brand_name'].notnull()].unique(), key = len, reverse = True)
brands_list = "\\b("+"|".join([re.escape(m) for m in brands_list])+")\\b"
print("Start fill brand : " + str(datetime.datetime.now().time()))
brands_from_name = parallelize(brand_extract, whole['name'][whole['brand_name'].isnull()].to_frame(), 'name', brands_list)
whole['brand_name'].fillna(brands_from_name, inplace=True)
print("End fill brand from name: " + str(datetime.datetime.now().time()))
brands_from_desc = parallelize(brand_extract, whole['item_description'][whole['brand_name'].isnull()].to_frame(), 'item_description', brands_list)
whole['brand_name'].fillna(brands_from_desc, inplace=True)
print("End fill brand from item_description: " + str(datetime.datetime.now().time()))
print(whole.count())

### last fills
fill_na_unique(whole, ['category_name','brand_name'])
print("End column creation and na fill : " + str(datetime.datetime.now().time()))

## Preprocessing categorical columns
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
csc_ship = csc_matrix(whole['shipping']).transpose()
csc_cond = csc_matrix(pd.get_dummies(whole['item_condition_id'], sparse = True))
#csc_ship_cond = whole[['shipping','item_condition_id']]
print("End shipping and condition preprocessing : " + str(datetime.datetime.now().time()))

## Final csc
csc_final = hstack((csc_desc, csc_name, csc_brand, csc_brand_SI, csc_cat, csc_cat_SI, csc_ship, csc_cond, csc_has_brand, csc_has_cat, csc_has_desc))
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
   df_test['predicted'] = np.exp(df_test['predicted'])-1
   df_test['predicted'][df_test['predicted'] < 3] = 3
   df_test['predicted'] = np.log(df_test['predicted']+1)
   df_test['eval'] = (df_test['predicted'] - df_test['price'])**2
   eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
   print("score: ", eval1)
else:
   df_sub = pd.DataFrame({'test_id':df_test.index})
   df_sub['price'] = np.exp(estimator.predict(csc_test))-1
   df_sub['price'][df_sub['price'] < 3] = 3
   df_sub.to_csv('submission.csv',index=False)
