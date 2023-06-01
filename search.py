from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
from nltk.corpus import wordnet
import nltk
import random
import tensorflow as tf
from keras.models import load_model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.layers import Input, Dense, Activation, TimeDistributed, Softmax, TextVectorization, Reshape, RepeatVector, Conv1D, Bidirectional, AveragePooling1D, UpSampling1D, Embedding, Concatenate, GlobalAveragePooling1D, LSTM, Multiply
from keras.models import Model
import tensorflow as tf
import keras

nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')


DATASET = 'crawler_data.csv'
df = pd.read_csv(DATASET)


vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['body'])
index = dict()

for w in tqdm(vectorizer.vocabulary_.keys()):
    index[w] = dict()
    for j in range(tfidf.shape[0]):
        if tfidf[j, vectorizer.vocabulary_[w]] > 0:
            index[w][j] = tfidf[j, vectorizer.vocabulary_[w]]


def buscar(palavras, indice):
    assert type(palavras)==list
    resultado = dict()
    for p in palavras:
        if p in indice.keys():
            for documento in indice[p].keys():
                if documento not in resultado.keys():
                    resultado[documento] = indice[p][documento]
                else:
                    resultado[documento] += indice[p][documento]
                
    return resultado

#2 
def n_relevantes(result_busca, n):
    res = []
    for key in result_busca.keys():
        res.append( (result_busca[key], key)) 

    res = sorted(res, reverse= True)[0 : n]

    return res

#3
def query(q_str, n, index):

    words = re.findall('\w+', q_str)
    res = buscar(words, index)
    res_n = n_relevantes(res, n)
    return res_n


def train(index = index):
    for w in tqdm(vectorizer.vocabulary_.keys()):
        index[w] = dict()
        for j in range(tfidf.shape[0]):
            if tfidf[j, vectorizer.vocabulary_[w]] > 0:
                index[w][j] = tfidf[j, vectorizer.vocabulary_[w]]


def tfidf_search(command):

    pattern = r"!search (.+)(?:\sth=(?:\d+(\.\d+))?)?"
    pattern2 = r"(.+)(?:\s)"
    pattern3 = r".*?th=(\d+(\.\d+)?)"


    groups = re.match(pattern, command)
    if groups:

        term = re.match(pattern2 , groups.group(1))
        if term is not None:
            term = term.group(1)

        threshold  = re.match(pattern3, groups.group(1))
        if threshold is not None:
            threshold = threshold.group(1)
            threshold = float(threshold)
    
    #aqui usamos tudo acima para pegar o documento com maior tf-idf, com indice invertido
    result = query(term, 1, index)
    print("result:")
    print(result)

    if result:
        # print(result[0][1])
        url = df.loc[result[0][1]].url
        content = df.loc[result[0][1]].body

        if threshold is not None:
            th = content_filter(content)
            if th < threshold:
                return "resultado abaixo do threshold especificado :("
            
        return (url, content)
    
    return "Nao Encontrado"


def wn_search(command):
    url = 'none'
    max_value = 0

    pattern = r"!wn_search (.+)(?:\sth=(?:\d+(\.\d+))?)?"
    pattern2 = r"(.+)(?:\s)"
    pattern3 = r".*?th=(\d+(\.\d+)?)"


    groups= re.match(pattern, command)
    if groups():

        term = re.match(pattern2 , groups.group(1))
        if term is not None:
            term = term.group(1)

        threshold  = re.match(pattern3, groups.group(1))
        if threshold is not None:
            threshold = threshold.group(1)
            threshold = float(threshold)

    synsets = wordnet.synsets(term, lang='por')
    print([syn for syn in synsets])
    print([syn.name() for syn in synsets])
    print([syn.definition() for syn in synsets])

    #aqui usamos tudo acima para pegar o documento com maior tf-idf, com indice invertido
    result = query(term, 1, index)
    # print(result)

    if result:
        url = df.loc[result[0][1]].url
        max_value = result[0][0]
   
    for syn in synsets:

        definition = syn.definition()
        result = query(definition, 1, index)

        if result:
            value = result[0][0]

            if value > max_value:
                url = df.loc[result[0][1]].url
                content = df.loc[result[0][1]].body


    if url != 'none':
        if threshold is not None:

            th = content_filter(content)

            if th < threshold:
                return "resultado acima do threshold especificado :("
            
        return (url, content)
    
    return "Nao Encontrado"


def content_filter(content):

    bad_words = 'datasets/bad_words.csv'
    good_words = 'datasets/words_pos.csv'

    bad_words = pd.read_csv(bad_words)
    good_words = pd.read_csv(good_words)


    good_words = good_words.drop(columns=['pos_tag'])
    good_words['IsBad'] = 0
    bad_words['IsBad'] = 1

    good_words_sample = good_words.sample(1618, random_state=42)

    words = pd.concat([good_words_sample, bad_words])

    X = words["word"]
    y = words["IsBad"]

    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)

    classificador = Pipeline([
                        ('meu_vetorizador', CountVectorizer(stop_words='english')),
                        ('meu_classificador', LogisticRegression(penalty=None, solver='saga', max_iter=10000))
                        ])
    
    classificador.fit(X_train,y_train)
    y_pred = classificador.predict(X_test)
    acc = accuracy_score(y_pred,y_test)

    prob = classificador.predict_log_proba([content])
    probas = classificador.predict_proba([content])

    if prob[0][1] >= 1:
        return 1
    elif prob[0][1] <= -1:
        return -1
    

    m = np.max(probas)
    prob = 2 * (m -prob[0][1]) / (2 * m) - 1

    return prob #[0][1]
