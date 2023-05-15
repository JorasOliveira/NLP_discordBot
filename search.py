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
    match = re.match(r"!search (.+)", command)

    term = match.group(1)
    
    #aqui usamos tudo acima para pegar o documento com maior tf-idf, com indice invertido
    result = query(term, 1, index)
    print(result)

    if result:
        print(result[0][1])
        url = df.loc[result[0][1]].url
        content = df.loc[result[0][1]].body
        print(url)
        return (url, content)
    
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
                content = df.loc[result[0][1]].body


    if url != 'none':
        print(url)
        return (url, content)
    
    return "Nao Encontrado"





#### APS 4 - geracao de conteudo ####


def predict_word(seq_len, latent_dim, vocab_size):
    input_layer = Input(shape=(seq_len-1,))
    x = input_layer
    x = Embedding(vocab_size, latent_dim, name='embedding', mask_zero=True)(x)
    x = LSTM(latent_dim, kernel_initializer='glorot_uniform')(x)
    latent_rep = x
    x = Dense(vocab_size)(x)
    x = Softmax()(x)
    return Model(input_layer, x), Model(input_layer, latent_rep)

vocab_size = 5000

def beam_search_predizer(entrada, numero_de_predicoes, modelo, vectorize_layer, t, beam_size):
    frase = entrada
    contexto = frase  # Contexto deslizante
    for n in range(numero_de_predicoes):
        if n == 0:
            pred = modelo.predict(vectorize_layer([contexto])[:, :-1])
            pred = pred * t

            # Select top candidates from the initial predictions
            top_tokens = np.argsort(pred[0, -1, :])[-beam_size:]
        else:
            new_candidates = []
            for token in top_tokens:
                new_context = contexto + " " + vectorize_layer.get_vocabulary()[token]
                pred = modelo.predict(vectorize_layer([new_context])[:, :-1])
                pred = pred * t
                new_candidates.append((token, pred[0, -1, token]))

            # Select top candidates from all the expanded predictions
            new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
            top_tokens = [candidate[0] for candidate in new_candidates[:beam_size]]

        # Randomly select a token from the top candidates
        rand_token = random.choice(top_tokens)
        word = vectorize_layer.get_vocabulary()[rand_token]

        frase = frase + " " + word
        contexto = contexto + " " + word
        contexto = ' '.join(contexto.split()[1:])
        print(word)

    return frase


def content_generator(command):
    
    match = re.match(r"!generate (.+)", command)

    term = match.group(1)

    texto =  tfidf_search("!search " + term)
    
    if texto == "Nao Encontrado":
        texto = wn_search("!wn_search " + term)

    texto = texto[1]

    # Ensure that texto is a list of strings
    if isinstance(texto, str):
        texto = [texto]
    elif isinstance(texto, tuple):
        texto = list(texto)

    print(texto)   

    predictor, latent = predict_word(10, 15, vocab_size)
    predictor.summary()
    #opt = keras.optimizers.SGD(learning_rate=1, momentum=0.9)
    opt = keras.optimizers.Nadam(learning_rate=0.1)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        ignore_class=1,
        name="sparse_categorical_crossentropy",
    )

    
    predictor.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])

    vectorize_layer = TextVectorization(max_tokens=vocab_size, output_sequence_length=10)

    vectorize_layer.adapt(texto)

    pred = predictor.predict(vectorize_layer([texto])[:,:-1])
    idx = tf.argmax(pred, axis=1)[0]
    idx = idx.numpy() - 1
    print(idx)
    vectorize_layer = vectorize_layer.get_vocabulary()[idx]

    num_predictions = 10
    temperature = 0.1
    beam_size = 5

    # Generate predictions using beam search
    output_text = beam_search_predizer(texto, num_predictions, pred, vectorize_layer, temperature, beam_size)

    return output_text
