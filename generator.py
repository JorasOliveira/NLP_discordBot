from keras.layers import Input, Dense, Activation, TimeDistributed, Softmax, TextVectorization, Reshape, RepeatVector, Conv1D, Bidirectional, AveragePooling1D, UpSampling1D, Embedding, Concatenate, GlobalAveragePooling1D, LSTM, Multiply
from keras.models import Model
import tensorflow as tf
import keras
import numpy as np
import random

import pandas as pd
import os
import re
from tqdm import tqdm

from keras.layers import Input, TextVectorization
from keras.models import Model
from transformers import pipeline, set_seed

DATASET = 'crawler_data.csv'
df = pd.read_csv(DATASET)

ds = tf.data.Dataset.from_tensor_slices(df)

vocab_size = 5000
vectorize_layer = TextVectorization(max_tokens=vocab_size, output_sequence_length=10)
vectorize_layer.adapt(ds)

def separar_ultimo_token(x):


    x_ = vectorize_layer(x)
    x_ = x_[:,:-1]
    y_ = x_[:,-1:]
    return x_, y_

def predict_word(seq_len, latent_dim, vocab_size):
    input_layer = Input(shape=(seq_len-1,))
    x = input_layer
    x = Embedding(vocab_size, latent_dim, name='embedding', mask_zero=True)(x)
    x = LSTM(latent_dim, kernel_initializer='glorot_uniform')(x)
    latent_rep = x
    x = Dense(vocab_size)(x)
    x = Softmax()(x)
    return Model(input_layer, x), Model(input_layer, latent_rep)


def beam_search_predizer(entrada, numero_de_predicoes, modelo, vectorize_layer, beam_size):
    frase = entrada
    contexto = frase  # Contexto deslizante
    for n in range(numero_de_predicoes):
        pred = modelo.predict(vectorize_layer([contexto])[:, :-1])

        # Beam search
        candidates = []
        for _ in range(beam_size):
            # Select top candidate from the predictions
            idx = np.argmax(pred, axis=1)[0]
            word = vectorize_layer.get_vocabulary()[idx]

            # Check if the word is already in the sentence
            if word not in frase.split():
                candidates.append((word, pred[0, idx]))

            # Set the probability of the selected word to 0
            pred[0, idx] = 0

        # Sort candidates based on probability in descending order
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select the top candidate as the next word
        word = candidates[0][0]

        frase = frase + " " + word
        contexto = contexto + " " + word
        contexto = ' '.join(contexto.split()[1:])
        print(word)

    return frase



predictor, latent = predict_word(10, 15, vocab_size)
predictor.summary()
#opt = keras.optimizers.SGD(learning_rate=1, momentum=0.9)
opt = keras.optimizers.Nadam(learning_rate=0.1)
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    ignore_class=1,
    name="sparse_categorical_crossentropy",
)

predictor.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])

tf.config.run_functions_eagerly(True)

history = predictor.fit(ds.map(separar_ultimo_token), epochs=20, verbose=1)


def content_generator(command):

    match = re.match(r"!generate (.+)", command)
    content = match.group(1)

    return beam_search_predizer(content, 15, predictor, vectorize_layer, beam_size=20)
    

def gpt2_generate(command):
    match = re.match(r"!gpt2_generate (.+)", command)
    content = match.group(1)

    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    gpt_output = generator(content, max_length=120, num_return_sequences=1)

    generated_text = gpt_output[0]['generated_text']  # Access the generated text from the output

    return generated_text

