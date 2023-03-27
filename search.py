from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
from nltk.corpus import wordnet
import nltk
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
    match = re.match(r"!search (.+)", command)

    term = match.group(1)
    #aqui usamos tudo acima para pegar o documento com maior tf-idf, com indice invertido
    result = query(term, 1, index)
    print(result)

    if result:
        print(result[0][1])
        url = df.loc[result[0][1]].url
        print(url)
        return url
    
    return "Nao Encontrado"


def wn_search(command):
    url = 'none'
    max_value = 0
    
    match = re.match(r"!wn_search (.+)", command)

    term = match.group(1)

    synsets = wordnet.synsets(term, lang='por')
    print([syn for syn in synsets])
    print([syn.name() for syn in synsets])
    print([syn.definition() for syn in synsets])

    #aqui usamos tudo acima para pegar o documento com maior tf-idf, com indice invertido
    result = query(term, 1, index)
    print(result)

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


    if url != 'none':
        print(url)
        return url
    
    return "Nao Encontrado"