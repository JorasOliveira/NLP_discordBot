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

from search import tfidf_search
from search import wn_search


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

    # # Ensure that texto is a list of strings
    # if isinstance(texto, str):
    #     texto = [texto]
    # elif isinstance(texto, tuple):
    #     texto = list(texto)

    print(texto[0])

    print('#####################################################################################################################')
    print(texto[1])   


    predictor, latent = predict_word(10, 15, vocab_size)
    predictor.summary()
    #opt = keras.optimizers.SGD(learning_rate=1, momentum=0.9)
    opt = keras.optimizers.Nadam(learning_rate=0.1)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        ignore_class=1,
        name="sparse_categorical_crossentropy",
    )

    vectorize_layer = TextVectorization(max_tokens=vocab_size, output_sequence_length=10)

    vectorize_layer.adapt(texto)
     
    predictor.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])
    
    predictor.fit(texto, epochs=20, verbose=1)

    
    # pred = predictor.predict(vectorize_layer([texto])[:,:-1])
    # # idx = tf.argmax(pred, axis=1)[0]
    # # idx = idx - 1
    # # print(idx)
    # # vectorize_layer = vectorize_layer.get_vocabulary()[idx]

    # num_predictions = 10
    # temperature = 0.1
    # beam_size = 5

    # # Generate predictions using beam search
    # output_text = beam_search_predizer(texto, num_predictions, pred, vectorize_layer, temperature, beam_size)

    # return output_text

