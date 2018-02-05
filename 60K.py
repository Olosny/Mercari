#!/usr/bin/python3

#############
## Imports ##
#############

## Obs
import datetime, time
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, RFE
## nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

## multiprocessing
from multiprocessing import cpu_count, Pool

## keras
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, GRU, concatenate, LSTM, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Dropout, AveragePooling1D
from keras.layers.merge import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams, pad_sequences
from keras import optimizers

## gensim
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple
    
################
## Parameters ##
################

## run sur train split ou sur test pour submit
SUB = False
SMALL = False

## Strat
STEMMING = False
HAS_BRAND = False
HAS_DESC = True
HAS_CAT = True
FILL_BRAND = True
HAS_BRAND_NOW = True
IS_BUNDLE = True
NLP_DESC = True
NLP_NAME = True
NLP_BRAND = False
NLP_CAT_SPLIT = False
NLP_CAT_UNSPLIT = False
HOT_BRAND = True
NAME_LEN = True
DESC_LEN = True
DESC_LEN_POLY = False
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
TOKEN_PAT = "(?u)\\b\w+\\b"
STEMMER = SnowballStemmer('english', ignore_stopwords = False)
STOP_W = set(stopwords.words('english'))
more_stop = set(['i\'d', 'i\'m', 'i\'ll', ';)', '***', '**', ':)', '(:', '(;', ':-)', '//'])
STOP_W |= more_stop

BUNDLE_RE = "\\b(bundl\w?|joblot|lot|2x|3x|4x)\\b"
###############
## Functions ##
###############

## -------------------- Neural Net ----------------------
def sanitize_text(df, col, pat):
    #chars_del = re.compile(r"[^A-Za-z0-9\n]")
    chars_del = re.compile(pat)
    #return df[col].str.lower().str.replace(chars_del, " ").apply(lambda x : ' '.join([word for word in str(x).split(' ') if word not in STOP_W]))
    return df[col].str.replace(chars_del, " ").apply(lambda x : ' '.join([word for word in str(x).split(' ') if str(word).lower() not in STOP_W]))

def get_nn_data(df):
    X = {'name': pad_sequences(df.name, padding = 'post'),
         'item_desc': pad_sequences([i[:ct_dict['desc_len']] for i in df.item_description], padding = 'post'),
         'brand_name': np.array(df[['brand_name']]),
         'item_condition': np.array(df[['item_condition_id']]),
         'shipping': np.array(df[["shipping"]]),
         'desc_len': np.array(df[["desc_len"]]),
         'name_len': np.array(df[["name_len"]]),
         'has_desc': np.array(df[['has_item_description']]),
         'has_cat': np.array(df[['has_category_name']]),
         'has_brand_now': np.array(df[['has_brand_now']]),
         'is_bundle': np.array(df[['is_bundle']])}
    
    for col in [i for i in whole.columns if 'category_name_' in i]:
        X.update({col : np.array(df[[col]])})
    
    return X

def good_model():
    # Inputs
    name = Input(shape=[X["name"].shape[1]], name="name")
    desc = Input(shape=[X["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[X["brand_name"].shape[1]], name="brand_name")
    cond = Input(shape=[X["item_condition"].shape[1]], name="item_condition")
    shipping = Input(shape=[X["shipping"].shape[1]], name="shipping")
    desc_len = Input(shape=[X["desc_len"].shape[1]], name="desc_len")
    name_len = Input(shape=[X["name_len"].shape[1]], name="name_len")
    cat_0 = Input(shape=[X["category_name_0"].shape[1]], name="category_name_0")
    cat_1 = Input(shape=[X["category_name_1"].shape[1]], name="category_name_1")
    cat_2 = Input(shape=[X["category_name_2"].shape[1]], name="category_name_2")
    cat_3 = Input(shape=[X["category_name_3"].shape[1]], name="category_name_3")
    cat_4 = Input(shape=[X["category_name_4"].shape[1]], name="category_name_4")
    has_desc = Input(shape=[X["has_desc"].shape[1]], name="has_desc")
    has_cat = Input(shape=[X["has_cat"].shape[1]], name="has_cat")
    has_brand_now = Input(shape=[X["has_brand_now"].shape[1]], name="has_brand_now")
    is_bundle = Input(shape=[X["is_bundle"].shape[1]], name="is_bundle")
    
    # Embeddings
    emb_name = Embedding(ct_dict['num_name'], 20)(name)
    emb_desc = Embedding(ct_dict['num_item_desc'], 50)(desc)
    emb_brand = Embedding(ct_dict['num_brand_name'], 10)(brand)
    emb_cond = Embedding(ct_dict['num_item_condition'], 5)(cond)
    #emb_desc_len = Embedding(ct_dict['num_desc_len'], 5)(desc_len)
    #emb_name_len = Embedding(ct_dict['num_name_len'], 5)(name_len)
    emb_cat_0 = Embedding(ct_dict['num_category_name_0'], 10)(cat_0)
    emb_cat_1 = Embedding(ct_dict['num_category_name_1'], 10)(cat_1)
    emb_cat_2 = Embedding(ct_dict['num_category_name_2'], 30)(cat_2)
    emb_cat_3 = Embedding(ct_dict['num_category_name_3'], 10)(cat_3)
    emb_cat_4 = Embedding(ct_dict['num_category_name_4'], 10)(cat_4)
    
    # Convnet
    conv_name = Conv1D(16, 9, activation = 'relu')(emb_name)
    pool_name = GlobalMaxPooling1D()(conv_name)
    conv_desc = Conv1D(16, 5, activation = 'relu')(emb_desc)
    pool_desc = GlobalMaxPooling1D()(conv_desc)
    #gru_name  = GRU(8)(emb_name)
    #gru_desc  = GRU(16)(emb_desc)
    #pool_name = GRU(8)(emb_name)
    #pool_desc = GRU(16)(emb_desc)
    
    # Concat
    x = concatenate([pool_name,
                     pool_desc,
                     gru_name,
                     gru_desc,
                     Flatten()(emb_brand),
                     Flatten()(emb_cond),
                     shipping,
                     name_len,
                     desc_len,
                     Flatten()(emb_cat_0),
                     Flatten()(emb_cat_1),
                     Flatten()(emb_cat_2),
                     Flatten()(emb_cat_3),
                     Flatten()(emb_cat_4),
                     has_desc,
                     has_cat,
                     has_brand_now,
                     is_bundle])
    #x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    
    # Dense
    x = Dense(512, activation = 'relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    #x = Dense(256, activation = 'relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    x = Dense(128, activation = 'relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    #x = Dense(64, activation = 'relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    
    # Output
    output = Dense(1, activation = 'linear')(x)
    
    # Model
    model = Model([name, desc, brand, cond, shipping, desc_len, name_len, cat_0, cat_1, cat_2, cat_3, cat_4, has_cat, has_desc, has_brand_now, is_bundle], output)
    model.compile(loss = 'mse', optimizer = 'adam')
    
    return model


def emb_model(cat, name, desc, ship_cond):
    cat_input = Input(shape = (cat.shape[1],))
    name_input = Input(shape = (name.shape[1],))
    #desc_input = Input(shape = (desc.shape[1],))
    x = Embedding(1025, 64, input_length=8)(cat_input)
    y = Embedding(100000, 64, input_length=17)(name_input)
    #z = Embedding(10000, 64, input_length=264)(desc_input)
    cat_lstm_out = LSTM(32)(x)
    name_lstm_out = LSTM(32)(y)
    #desc_lstm_out = LSTM(16)(z)
    cat_output = Dense(1, activation='sigmoid')(cat_lstm_out)
    name_output = Dense(1, activation='sigmoid')(name_lstm_out)
    #desc_output = Dense(1, activation='sigmoid')(desc_lstm_out)
    easy_input = Input(shape = (ship_cond.shape[1],))
    #x = concatenate([cat_lstm_out, name_lstm_out, desc_lstm_out, easy_input])
    x = concatenate([cat_lstm_out, name_lstm_out, easy_input])
    x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    main_output = Dense(1, activation = 'relu')(x)
    #model = Model(inputs=[cat_input, name_input, desc_input, easy_input], outputs=[main_output, cat_output, name_output, desc_output])
    model = Model(inputs=[cat_input, name_input, easy_input], outputs=[main_output, cat_output, name_output])
    #sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #rms = optimizers.RMSprop(lr=0.002)
    model.compile(optimizer='rmsprop', loss=['mse', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[1., 0.1, 0.1], metrics = ['mae'])
    #model.compile(optimizer='rmsprop', loss='mse', loss_weights=[1., 0.1, 0.1, 0.1], metrics = ['mae'])
    #model.compile(optimizer='rmsprop', loss='mse', loss_weights=[1., 0.1, 0.1], metrics = ['mae'])
    #model = Sequential()
    #model.add(Embedding(1025, 64, input_length=8))
    #model.add(Flatten())
    #model.add(Dense(1, activation = 'relu'))
    #model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def conv_emb_model(cat, name, desc, ship_cond):
    cat_input = Input(shape = (cat.shape[1],))
    name_input = Input(shape = (name.shape[1],))
    desc_input = Input(shape = (desc.shape[1],))
    x = Embedding(max([max(i) for i in cat]) + 1, 64, input_length=11)(cat_input)
    y = Embedding(max([max(i) for i in name]) + 1, 128, input_length=17)(name_input)
    z = Embedding(max([max(i) for i in desc]) + 1, 64, input_length=50)(desc_input)
    cat_lstm_out = Conv1D(32, 7, activation = 'relu')(x)
    cat_lstm_out = GlobalMaxPooling1D()(cat_lstm_out)
    #cat_lstm_out_1 = Conv1D(32, 7, activation = 'relu')(x)
    #cat_lstm_out_1 = GlobalMaxPooling1D()(cat_lstm_out_1)
    #cat_lstm_out_2 = Conv1D(32, 3, activation = 'relu')(x)
    #cat_lstm_out_2 = GlobalMaxPooling1D()(cat_lstm_out_2)
    name_lstm_out = Conv1D(32, 9, activation = 'relu')(y)
    name_lstm_out = GlobalMaxPooling1D()(name_lstm_out)
    #name_lstm_out_1 = Conv1D(32, 9, activation = 'relu')(y)
    #name_lstm_out_1 = GlobalMaxPooling1D()(name_lstm_out_1)
    #name_lstm_out_2 = Conv1D(32, 3, activation = 'relu')(y)
    #name_lstm_out_2 = GlobalMaxPooling1D()(name_lstm_out_2)
    desc_lstm_out = Conv1D(32, 5, activation = 'relu')(z)
    desc_lstm_out = GlobalMaxPooling1D()(desc_lstm_out)
    #desc_lstm_out_1 = Conv1D(32, 5, activation = 'relu')(z)
    #desc_lstm_out_1 = GlobalMaxPooling1D()(desc_lstm_out_1)
    #desc_lstm_out_2 = Conv1D(32, 3, activation = 'relu')(z)
    #desc_lstm_out_2 = GlobalMaxPooling1D()(desc_lstm_out_2)
    #cat_lstm_out = concatenate([cat_lstm_out_1, cat_lstm_out_2])
    #name_lstm_out = concatenate([name_lstm_out_1, name_lstm_out_2])
    #desc_lstm_out = concatenate([desc_lstm_out_1, desc_lstm_out_2])
    cat_output = Dense(1, activation='sigmoid')(cat_lstm_out)
    name_output = Dense(1, activation='sigmoid')(name_lstm_out)
    desc_output = Dense(1, activation='sigmoid')(desc_lstm_out)
    easy_input = Input(shape = (ship_cond.shape[1],))
    x = concatenate([cat_lstm_out, name_lstm_out, desc_lstm_out, easy_input])
    #x = concatenate([cat_lstm_out, desc_lstm_out, easy_input])
    #x = concatenate([cat_lstm_out, name_lstm_out, easy_input])
    x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    main_output = Dense(1, activation = 'relu')(x)
    model = Model(inputs=[cat_input, name_input, desc_input, easy_input], outputs=[main_output, cat_output, name_output, desc_output])
    #model = Model(inputs=[cat_input, desc_input, easy_input], outputs=[main_output, cat_output, desc_output])
    #model = Model(inputs=[cat_input, name_input, easy_input], outputs=[main_output, cat_output, name_output])
    #sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #rms = optimizers.RMSprop(lr=0.002)
    model.compile(optimizer='rmsprop', loss=['mse', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[1., 0.01, 0.01, 0.01], metrics = ['mae'])
    #model.compile(optimizer='rmsprop', loss=['mse', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[1., 0.1, 0.1], metrics = ['mae'])
    #model.compile(optimizer='rmsprop', loss=['mse', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[1., 0.1, 0.1], metrics = ['mae'])
    #model.compile(optimizer='rmsprop', loss='mse', loss_weights=[1., 0.1, 0.1, 0.1], metrics = ['mae'])
    #model.compile(optimizer='rmsprop', loss='mse', loss_weights=[1., 0.1, 0.1], metrics = ['mae'])
    #model = Sequential()
    #model.add(Embedding(1025, 64, input_length=8))
    #model.add(Flatten())
    #model.add(Dense(1, activation = 'relu'))
    #model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def gen_model(name_s, ship_cond_s):
    name_input = Input(shape = (name_s, ))
    ship_cond_input = Input(shape = (ship_cond_s, ))
    x = concatenate([name_input, ship_cond_input])
    x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    main_output = Dense(1, activation = 'relu')(x)
    model.compile(optimizer='rmsprop', loss='mse', metrics = ['mae'])
    return model

def lol_model():
    model = Sequential()
    model.add(Embedding(10000, 64, input_length=17))
    model.add(Flatten())
    model.add(Dense(1, activation = 'relu'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def build_model(shape):
    model = Sequential()
    #model.add(Dense(shape / 4, activation='relu', input_shape=(shape, )))
    model.add(Dense(512, activation='relu', input_shape=(shape, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation = 'relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#def tokenize(df, col, max_word):
#    vectorizer = TfidfVectorizer(strip_accents = 'unicode',
#                             token_pattern = TOKEN_PAT,
#                             stop_words = STOP_W,
#                             max_features = max_word,
#                             smooth_idf = True,
#                             sublinear_tf = False)                                 
#    analyzer = vectorizer.build_analyzer()
#    #tokenizer.fit(df[col])
#    vectorizer.fit(df[col])
#    output_corpus = [list(map(lambda x: vectorizer.vocabulary_.get(x, max_word) + 1, analyzer(line))) for line in df[col]]
#    #return [[tokenizer.vocabulary_[j] for j in k] for k in [i for i in words]]
#    return output_corpus
#    #tokenizer = Tokenizer(num_words = max_word)
#    #tokenizer = Tokenizer(num_words = max_word)
#    #tokenizer.fit_on_texts(df[col])
#    #return tokenizer.texts_to_sequences(df[col])

def tokenize(df, col, max_word):
    tokenizer = Tokenizer(num_words = max_word, filters = '', lower = False)
    tokenizer.fit_on_texts(df[col])
    return tokenizer.texts_to_sequences(df[col])

def enc_d2v(df, col, min_count = 2, window = 5, size = 100):
    token_col = df[col].apply(CountVectorizer().build_analyzer())
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    
    for i, text in enumerate(token_col):
        words = text
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    
    print("Start build_vocab")
    #d2v = Doc2Vec(docs, min_count=min_count, window=window, size=size, workers = 3, iter = 2, dm = 0)
    d2v = Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=min_count, window=window, size=size, workers = 3, dm = 0)
    d2v.build_vocab(docs)
    for epoch in range(2):
        print(" Start epoch : " + str(epoch + 1))
        d2v.train(docs, total_examples=d2v.corpus_count, epochs = d2v.iter)
        d2v.alpha -= 0.002  # decrease the learning rate
        d2v.min_alpha = d2v.alpha  # fix the learning rate, no decay
    
    vects = np.zeros((df[col].size, d2v.vector_size))
    
    for i in range(df[col].size):
        vects[i] = d2v.docvecs[i]
    
    d2v.delete_temporary_training_data()
    return vects

def sentence_vect(sentence, dic, wv, size):
    average_vect = [0 for i in range(size)]
    
    for word in sentence:
        if word in wv.vocab:
            idf = dic.get(word) if dic is not None else 1
            v = [idf * value for value in wv[word]]
            average_vect = map(sum, zip(average_vect,v))
    
    if len(sentence) != 0:
        average_vect = [val/len(sentence) for val in average_vect]
    
    return average_vect

def enc_w2v(df, col, idf = True, min_count = 2, window = 5, size = 100):
    vectorizer = TfidfVectorizer()                                
     
    vectorizer.fit(df[col])
    dic = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)) if idf else None
    
    token_col = df[col].apply(vectorizer.build_analyzer())
    w2v = Word2Vec(token_col, min_count=min_count,  window=window, size=size, workers=4)
    
    sentence_vects = token_col.apply(lambda x: sentence_vect(x, dic, w2v.wv, size))
    return pd.DataFrame.from_items(zip(sentence_vects.index, sentence_vects.values)).T

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
    print("    Begin "+col+" NLP : ")
    t1 = time.time()
    
    df_col = df[col]
    if STEMMING:
        df_col = parallelize(stem, df_col.to_frame(), col, STEMMER)
    
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
    print("    shape : ", my_doc_term.shape)
    
    print("    End "+col+" NLP : " + str(time.time()-t1))
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
    return len([w for w in CountVectorizer(token_pattern=pat).build_analyzer()(str(doc))])

### Count words in col of df
def get_len(df, col, pat):
    return df[col].apply(lambda doc: word_count(doc, pat))

### Add feature word count for col of df
def len_feature(df, col, name, pat=TOKEN_PAT):
    new_col = parallelize(get_len, df, col, pat)
    df[name] = MaxAbsScaler().fit_transform(new_col.to_frame())

def len_feature_nonorm(df, col, name, pat=TOKEN_PAT):
    new_col = parallelize(get_len, df, col, pat)
    df[name] = new_col.to_frame()

###
def add_len(df, col, p=1):
    return csc_matrix(df[col].pow(p)).transpose()

### len_to_dummies()

##########
## Main ##
##########

## -------------------- Read and clean --------------------
print("Read and Pre-Preprocessing")
t = time.time()

### Read train
df_train = pd.read_table('../input/train.tsv', index_col = 0)            
sup_index_train = len(df_train)
df_train = df_train[df_train['price'] != 0]                    # drop price == 0$
df_train['price'] = np.log(df_train['price']+1)                # Price -> Log

### Read test
if SUB:
    df_test = pd.read_table('../input/test.tsv', index_col = 0)
    df_test.index = df_test.index + sup_index_train 
elif SMALL:
    drop_indices = np.random.choice(df_train.index, int(df_train.shape[0]*0.9), replace=False)
    df_train.drop(drop_indices, inplace=True)
    df_test = None
else:
    df_train, df_test = train_test_split(df_train, test_size=0.05, random_state=RAND)

split_index = len(df_train) 

print("Finished Pre-Preprocessing : " + str(time.time() - t) + "\n")


## -------------------- Preprocessing --------------------
print("Begin Preprocessing")
whole = pd.concat([df_train, df_test])

### Standardize text and Nans
print("  Begin standardization")
t = time.time()

str_to_nan(whole, 'item_description', 'No description yet')
#whole = whole.applymap(lambda x: x if type(x)!=str else x.lower())

print("  End standardization : " + str(time.time() - t))

csc_m_list=[]

### Feature creation
print("  Begin feature creation")
t = time.time()

if HAS_BRAND:
    csc_m_list.append(has_feature(whole, 'brand_name', 'has_brand'))

if HAS_DESC:
    csc_m_list.append(has_feature(whole, 'item_description', 'has_item_description'))

if HAS_CAT:
    csc_m_list.append(has_feature(whole, 'category_name', 'has_category_name'))

len_feature(whole, 'name', 'name_len')
len_feature(whole, 'item_description', 'desc_len')
whole['name'] = parallelize(sanitize_text, whole, 'name', r"[^A-Za-z0-9\n]")
whole['item_description'] = parallelize(sanitize_text, whole, 'item_description', r"[^A-Za-z0-9\n]")

print("  End feature creation : " + str(time.time() - t))

### Fill NaNs 1
fill_na_fast(whole, ['item_description','name'])
cats_list = split_col(whole, 'category_name', delim = '/')
fill_na_fast(whole, ['category_name']+cats_list)

### Extract some missing brand names
print("  Begin extracting brand names")
t = time.time()

if FILL_BRAND:
    brands_list = sorted(whole['brand_name'][whole['brand_name'].notnull()].unique(), key = len, reverse = True)
    brands_list = "\\b("+"|".join([re.escape(m) for m in brands_list])+")\\b"
    fill_from_extr(whole, 'brand_name', 'name', brands_list)
    fill_from_extr(whole, 'brand_name', 'item_description', brands_list)

if HAS_BRAND_NOW:
    csc_m_list.append(has_feature(whole, 'brand_name', 'has_brand_now'))

print("  End extracting brand names : " + str(time.time() - t))

### Fill NaNs 2
fill_na_fast(whole, 'brand_name')

### NLP
print("  Begin NLP Stuff")
t = time.time()

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

print("  End NLP Stuff : " + str(time.time() - t))

### Length Features
print("  Begin word counts")
t = time.time()

if NAME_LEN:
    #len_feature(whole, 'name', 'name_len')
    csc_m_list.append(add_len(whole, 'name_len'))

if DESC_LEN:
    #len_feature(whole, 'item_description', 'item_description_len')
    #csc_m_list.append(add_len(whole, 'item_description_len'))
    csc_m_list.append(add_len(whole, 'desc_len'))

if DESC_LEN_POLY:
    csc_m_list.append(add_len(whole, 'item_description_len', p=2))

print("  End word counts : " + str(time.time() - t))

### Dummies
print("  Begin shipping, category and condition processing")
t = time.time()

if IS_BUNDLE:
    whole['is_bundle'] = whole['item_description'].str.contains(BUNDLE_RE, case = False) | whole['name'].str.contains(BUNDLE_RE, case = False)
    csc_m_list.append(csc_matrix(whole['is_bundle']).transpose())

if SHIPPING:
    csc_m_list.append(csc_matrix(whole['shipping']).transpose())

if CONDITION:
    csc_m_list.append(csc_matrix(pd.get_dummies(whole['item_condition_id'], sparse = True)))

if MULTILAB_CAT:
    csc_m_list.append(multilabel(whole, 'category_name', '/'))

print("  End shipping, category and condition processing : " + str(time.time() - t))

## -------------------- Final csc --------------------
csc_final = hstack(csc_m_list)
print("-> csc_final shape : " + str(csc_final.shape) + " <-")
csc_train = csc_final.tocsr()[:split_index]
csc_test = csc_final.tocsr()[split_index:]

#estimator = Ridge(copy_X = False, solver="sag", fit_intercept=True, random_state=RAND, alpha = 1.5)
#selector = RFE(estimator, int(csc_final.shape[1]*0.9), step=0.034, verbose=2)
#csc_train = selector.fit_transform(csc_train, df_train.price)
#print("csc_trian shape: " + str(csc_train.shape))
#csc_test = selector.transform(csc_test)
#print("csc_test shape: " + str(csc_test.shape))

print("End Preprocessing\n")
print("Begin Ridge estimating\n")
t = time.time()
estimator = Ridge(copy_X = True, solver="sag", fit_intercept=True, random_state=RAND, alpha = 1.5)

print("estimator: ", estimator.__class__.__name__)
print("params: ", estimator.get_params())

estimator.fit(csc_train, df_train.price)
df_test['predicted_ri'] = estimator.predict(csc_test)
df_test['predicted_ri'] = np.exp(df_test['predicted_ri'])-1
df_test['predicted_ri'][df_test['predicted_ri'] < 3] = 3
df_test['predicted_ri'] = np.log(df_test['predicted_ri']+1)
df_test['eval'] = (df_test['predicted_ri'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)

print('End : ' + str(datetime.datetime.now().time()))

print("Begin w2v for Random Forest")
t = time.time()
word2vec_rf = pd.merge(enc_w2v(whole, 'name', size = 20), enc_w2v(whole, 'item_description', size = 10), right_index = True, left_index = True)
print('End : ' + str(datetime.datetime.now().time()))

print("Categorical to Label")
t = time.time()
le = LabelEncoder()
for col in [i for i in whole.columns if 'category_name_' in i or 'brand_name' in i ]:
    whole[col] = le.fit_transform(whole[col])

print('End : ' + str(time.time() - t))

print("Begin Random Forest estimating")
estimator = RandomForestRegressor(n_estimators=20, n_jobs=N_CORES ,verbose=1)

print("estimator: ", estimator.__class__.__name__)
print("params: ", estimator.get_params())

estimator.fit(pd.merge(whole.drop(['name', 'item_description', 'category_name', 'brand_name', 'price'], axis = 1)[:split_index],
                       word2vec_rf[:split_index],
                       left_index = True,
                       right_index = True),
              whole.price[:split_index])
df_test['predicted_rf'] = estimator.predict(pd.merge(whole.drop(['name', 'item_description', 'category_name', 'brand_name', 'price'], axis = 1)[split_index:],
                                                     word2vec_rf[split_index:],
                                                     left_index = True,
                                                     right_index = True))
df_test['predicted_rf'] = np.exp(df_test['predicted_rf'])-1
df_test['predicted_rf'][df_test['predicted_rf'] < 3] = 3
df_test['predicted_rf'] = np.log(df_test['predicted_rf']+1)
df_test['eval'] = (df_test['predicted_rf'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)

print('End : ' + str(datetime.datetime.now().time()))

print("Begin Deep Learning")
print('Prepare for NN input')
t = time.time()

whole['name'] = tokenize(whole, 'name', None)
whole['item_description'] = tokenize(whole, 'item_description', None)

ct_dict = {'desc_len' : 50}
           #'num_cond' : whole.item_condition_id.max() + 2,
           #'shipping' : whole.shipping.max() + 2,
           #'num_desc_len' : np.max(whole.desc_len.max()) + 2,
           #'num_name_len' : np.max(whole.name_len.max()) + 2,
           #'num_name' : np.max(whole.name.max()) + 2,
           #'num_desc' : np.max(whole.item_description.max()) + 2 }

#for col in [i for i in whole.columns if 'category_name_' in i or 'brand_name' in i]:
#    ct_dict.update({col : whole[col].max() + 2 })

X = get_nn_data(whole)
ct_dict.update({'num_' + k : np.max(X[k]) + 1 for k in X.keys()})
print('End : ' + str(time.time() - t))

print('Model DL')
t = time.time()

X_train = {k : X[k][:split_index] for k in X.keys()}
X_val = {k : X[k][split_index:] for k in X.keys()}

model = good_model()
history = model.fit(X_train,
                    whole.price[:split_index],
                    epochs = 2,
                    batch_size = 3000,
                    validation_data = (X_val, whole.price[split_index:]))

print('End : ' + str(time.time() - t))

print('Begin RNN estimating')
df_test['predicted_cnn'] = model.predict(X_val)
df_test['predicted_cnn'] = np.exp(df_test['predicted_cnn'])-1
df_test['predicted_cnn'][df_test['predicted_cnn'] < 3] = 3
df_test['predicted_cnn'] = np.log(df_test['predicted_cnn']+1)
df_test['eval'] = (df_test['predicted_cnn'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)
print('End : ' + str(datetime.datetime.now().time()))

print('Begin final estimating')
model = build_model(3)
model.fit(df_test[['predicted_ri', 'predicted_rf', 'predicted_cnn']], df_test.price, epochs = 3, batch_size = 3000)
df_test['predicted'] = model.predict(df_test[['predicted_ri', 'predicted_rf', 'predicted_cnn']])
df_test['predicted'] = np.exp(df_test['predicted'])-1
df_test['predicted'][df_test['predicted'] < 3] = 3
df_test['predicted'] = np.log(df_test['predicted']+1)
df_test['eval'] = (df_test['predicted'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)
print('End : ' + str(datetime.datetime.now().time()))
#whole['cat_brand'] = whole['category_name'] + '/' + whole['brand_name']
#cat_tok = tokenize(whole, 'cat_brand', 10000)
##cat_tok = tokenize(whole, 'category_name', 1025)
#cat_tok = pad_sequences(cat_tok, padding = 'post')
#name_tok = tokenize(whole, 'name', 10000)
#name_tok = pad_sequences(name_tok, padding = 'post')
#desc_tok = [i[:50] for i in tokenize(whole, 'item_description', 10000)]
#desc_tok = pad_sequences(desc_tok, padding = 'post')
#
##model = emb_model(cat_tok, name_tok, desc_tok, csc_train)
#model = conv_emb_model(cat_tok, name_tok, desc_tok, csc_train)
#history = model.fit([cat_tok[:split_index], name_tok[:split_index], desc_tok[:split_index], csc_train.toarray()],
#                    [df_train.price, df_train.price, df_train.price, df_train.price],
#                    epochs=5,
#                    batch_size=1000,
#                    validation_data=([cat_tok[split_index:], name_tok[split_index:], desc_tok[split_index:], csc_test.toarray()],
#                                     [df_test.price, df_test.price, df_test.price, df_test.price]))
#history = model.fit([cat_tok[:split_index], desc_tok[:split_index], csc_train.toarray()],
#                    [df_train.price, df_train.price, df_train.price],
#                    epochs=5,
#                    batch_size=2000,
#                    validation_data=([cat_tok[split_index:], desc_tok[split_index:], csc_test.toarray()],
#                                     [df_test.price, df_test.price, df_test.price]))
#history = model.fit([cat_tok[:split_index], name_tok[:split_index], csc_train.toarray()],
#                    [df_train.price, df_train.price, df_train.price],
#                    epochs=5,
#                    batch_size=1000,
#                    validation_data=([cat_tok[split_index:], name_tok[split_index:], csc_test.toarray()],
#                                     [df_test.price, df_test.price, df_test.price]))
#history = model.fit(csc_train.toarray(), df_train.price, epochs=2, batch_size=100, validation_data=(csc_test.toarray(), df_test.price))
#history = model.fit(csc_train.toarray(), df_train.price, epochs=2, batch_size=100)

#import matplotlib.pyplot as plt
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()
#
#plt.clf()
#mae = history.history['mean_absolute_error']
#val_mae = history.history['val_mean_absolute_error']
#plt.plot(epochs, mae, 'b', label='mae')
#plt.plot(epochs, val_mae, 'b', label='Validation mae')
#plt.title('Training and validation mae')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

#if not SUB:
#    #df_test['predicted'] = model.predict(csc_test)
#    #df_test['predicted'] = model.predict([cat_tok[1037162:], name_tok[1037162:], desc_tok[1037162:], csc_test.toarray()])[0]
#    df_test['predicted_cnn'] = model.predict(X_val)
#    #df_test['predicted_cnn'] = model.predict([cat_tok[1037162:], name_tok[1037162:], csc_test.toarray()])[0]
#    df_test['predicted_cnn'] = np.exp(df_test['predicted_cnn'])-1
#    df_test['predicted_cnn'][df_test['predicted_cnn'] < 3] = 3
#    df_test['predicted_cnn'] = np.log(df_test['predicted_cnn']+1)
#    df_test['eval'] = (df_test['predicted_cnn'] - df_test['price'])**2
#    eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
#    print("score: ", eval1)
#    print('End : ' + str(datetime.datetime.now().time()))
#else:
#    df_sub = pd.DataFrame({'test_id' : df_test.index - sup_index_train})
#    df_sub['price'] = np.exp(model.predict(csc_test.toarray()))-1
#    df_sub['price'][df_sub['price'] < 3] = 3
#    df_sub.to_csv('submission.csv',index=False)

'''
df_test['predicted_rf'] = np.exp(df_test['predicted_rf'])-1
df_test['predicted_rf'][df_test['predicted_rf'] < 3] = 3
df_test['predicted_rf'] = np.log(df_test['predicted_rf']+1)
df_test['eval'] = (df_test['predicted_rf'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)
df_test['predicted_cnn'] = np.exp(df_test['predicted_cnn'])-1
df_test['predicted_cnn'][df_test['predicted_cnn'] < 3] = 3
df_test['predicted_cnn'] = np.log(df_test['predicted_cnn']+1)
df_test['eval'] = (df_test['predicted_cnn'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)
df_test['predicted'] = np.exp(df_test['predicted'])-1
df_test['predicted'][df_test['predicted'] < 3] = 3
df_test['predicted'] = np.log(df_test['predicted']+1)
df_test['eval'] = (df_test['predicted'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)
df_test['predicted_rf'] = np.exp(df_test['predicted_rf'])-1
df_test['predicted_rf'][df_test['predicted_rf'] < 3] = 3
df_test['predicted_rf'] = np.log(df_test['predicted_rf']+1)
df_test['eval'] = (df_test['predicted_rf'] - df_test['price'])**2
eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
print("score: ", eval1)
df_love['stacked_rf'] = np.exp(df_love['stacked_rf'])-1
df_love['stacked_rf'][df_love['stacked_rf'] < 3] = 3
df_love['stacked_rf'] = np.log(df_love['stacked_rf']+1)
df_love['eval'] = (df_love['stacked_rf'] - df_love['price'])**2
eval1 = np.sqrt(1 / len(df_love['eval']) * df_love['eval'].sum())
print("score: ", eval1)
################
## Regression ##
################

#estimator = RandomForestRegressor(n_estimators=20, n_jobs=-1,verbose=1)
#estimator = ExtraTreesRegressor(n_estimators=20, n_jobs=-1,verbose=1)
#estimator = Ridge(copy_X = False, solver="sag", fit_intercept=True, random_state=RAND, alpha = 1.5)
estimator = Ridge(copy_X = True, solver="sag", fit_intercept=True, random_state=RAND, alpha = 1.5)
#estimator = SGDRegressor(loss='squared_epsilon_insensitive', penalty='l2', alpha=0.00001, max_iter=400, tol=None, epsilon=0.1, random_state=RAND, learning_rate='invscaling', eta0=0.01, power_t=0.15)

print("estimator: ", estimator.__class__.__name__)
print("params: ", estimator.get_params())

if SMALL:
    print(cross_val_score(estimator, csc_train, df_train.price, cv=5, verbose=2).mean())
else:
    estimator.fit(csc_train, df_train.price)
    if not SUB:
        df_test['predicted_ri'] = estimator.predict(csc_test)
        df_test['predicted_ri'] = np.exp(df_test['predicted_ri'])-1
        df_test['predicted_ri'][df_test['predicted_ri'] < 3] = 3
        df_test['predicted_ri'] = np.log(df_test['predicted_ri']+1)
        df_test['eval'] = (df_test['predicted_ri'] - df_test['price'])**2
        eval1 = np.sqrt(1 / len(df_test['eval']) * df_test['eval'].sum())
        print("score: ", eval1)
        print('End : ' + str(datetime.datetime.now().time()))
    else:
        df_sub = pd.DataFrame({'test_id' : df_test.index - sup_index_train})
        df_sub['price'] = np.exp(estimator.predict(csc_test))-1
        df_sub['price'][df_sub['price'] < 3] = 3
        df_sub.to_csv('submission.csv',index=False)
        '''
