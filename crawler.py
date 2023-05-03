# para trabalhar com diretórios / sistema operacional
import os
# para nos comunicarmos com a Web
import requests
# para extrair informações de páginas HTML
import bs4
from bs4 import BeautifulSoup
# utilizada para nos indicar o caminho do executável do Python
import sys
# Para criar um Data Frame
import pandas as pd
# Controlar espera entre requisições
import time
# Gerar valores aleatórios
import random
# Produto cartesiano
from sklearn.utils.extmath import cartesian
# Renderizar HTML
import IPython
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.layers import Input, Concatenate, Dense, Activation, TextVectorization, Embedding, GlobalAveragePooling1D, Conv1D, AveragePooling1D, Reshape, GRU, LSTM
from keras.models import Model
import tensorflow as tf


vocab_size = 1000


def crawl(command, max_files = 20): #recebe o commando do chatbot, pega a url, faz a sopa com a panela, dai adiciona a sopa no final do csv, faz isso para todos os links encontrados ate o contador estourar
    files = 0

    print(command[0])
    print(command[1])

    if command[1] == 0:
        #tratando a url:
        match = re.match(r"!crawl (.+)", command[0])

        if match:
            url = match.group(1)
        
    if command[1] == 1:
        #recebendo a url e fazendo o request:
        url = command[0]

    title, body, links = soup_pot(url)

    #Aqui poderia ser um try-catch, ou talvez alguma outra solucao mais elegante que checa antes se o link comeca com "https:://" ou sei la oque
    #porem eu nao estou com tempo suficiente para fazer algo bonito, entao temos isso, que resolve o problema.
    if ( (title == 0) or (body == 0) or (links == 0) ):
        print("erro para pegar informacoes do site!")

    else:
        #exrtaindo os conteudos dos links recursivamente:

        print(files)
        append_csv(url, title, body)

        #extraindo os links
        for link in links:
            # Check if the URL already exists in the CSV file

            if os.path.isfile(f'crawler_data.csv'):
                existing_df = pd.read_csv('crawler_data.csv')
                if link in existing_df['url'].values:
                    continue
                
                else:
                    title, body, more_links = soup_pot(link)
                    if ( (title == 0) or (body == 0) or (links == 0) ):
                        print("erro para pegar informacoes do site!")
                        
                    else:
                        links =  links + more_links

                        if files >= max_files:
                            return
                        print(f'adding: {link}, {title}, {body}' )
                        print(f'files downloaded: {files}')
                        append_csv(link, title, body)
                        files += 1


def soup_pot(url): #funcao que faz a sopa e retorna o prato feito
    #pegando os conteudos do site
    web_page = requests.get(url)
    web_page.encoding = 'utf-8'
        #usando o beufifull soup para tratar a pagina
    soup = BeautifulSoup(web_page.content, 'html.parser')

    # Get the title and body of the page
    try:
        title = soup.title.string.strip()
    except AttributeError:
        return (0,0,0)
    try:
        body = soup.get_text().strip()
    except AttributeError:
        return (0,0,0)

    #pegando os links e colocando em uma lista:
    links = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href').startswith('https://')]

    return (title, body, links)

def append_csv(url, title, body): #adiciona no final do csv o conteudo, incrementa o contador de arquivos salvos
    #salvando o conteudo em um csv:
    # Create a DataFrame with the data
    df = pd.DataFrame({'url': [url], 'title': [title], 'body': [body]})

    # Save the DataFrame to a CSV file
    with open('crawler_data.csv', mode='a', newline='') as file:
        df.to_csv(file, header=not file.tell(), index=False)


def bad_word_trainer():

    #TODO - continuar alterando para funcionar do jeito certo
    #usar o predict_proba para pegar a probabilidade de ser um bad word e salvar ela em uma coluna do csv do crawler
    df = pd.read_csv('datasets/bad-words.csv')#.sample(1000) #so alterei ate aqui    
    ohe = OneHotEncoder()
    y_ohe = ohe.fit_transform(df['sentiment'].to_numpy().reshape((-1,1))).todense()
    X_train, X_test, y_train, y_test = train_test_split(df['review'], y_ohe)

    vectorize_layer = TextVectorization(output_mode='int', max_tokens=vocab_size, pad_to_max_tokens=True, output_sequence_length=256)
    vectorize_layer.adapt(X_train)
    clf = cnn_embedding_softmax_model(vectorize_layer)
    print(clf.summary())
    clf.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = clf.fit(X_train, y_train, epochs=5, verbose=1, validation_split=0.1)
    clf.evaluate(X_test, y_test)

    vectorize_layer = TextVectorization(output_mode='int', max_tokens=vocab_size, pad_to_max_tokens=True, output_sequence_length=256)
    vectorize_layer.adapt(X_train)
    clf2 = cnn_embedding_softmax_model(vectorize_layer)

    print(clf2.summary())
    clf2.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = clf2.fit(X_train, y_train, epochs=5, verbose=1, validation_split=0.1)
    clf2.evaluate(X_test, y_test)

    clf1 = blstm_softmax_model(clf2)

    print(clf1.summary())
    clf.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = clf.fit(X_train, y_train, epochs=60, verbose=1, validation_split=0.1)
    clf.evaluate(X_test, y_test)

    clf0 = avg_embedding_softmax_model(clf1)

    print(clf0.summary())
    clf.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = clf.fit(X_train, y_train, epochs=60, verbose=1) # validation_split=0.1
    clf.evaluate(X_test, y_test)


def convolve_and_downsample(input_n_samples, input_embedding_size, n_filters, kernel_size=3, **kwargs): #camada costumizada, convolucao com downsampling
    input_layer = Input(shape=(input_n_samples,input_embedding_size)) 
    x = input_layer
    x = Conv1D( filters=n_filters, #o numero de filtros que voce quer aplicar, dita a dimensao da sua saida.
                kernel_size=kernel_size, #tamanho do seu filtro h da convolucao 
                padding='same',
                use_bias=False,
                )(x) #convolucao
    x = AveragePooling1D(pool_size=2)(x) #downsampling, poolsize 2 significa que voce vai pegar o valor medio a cada 2 elementos.
    x = Activation('elu')(x) #funcao de ativação
    return Model(input_layer, x, **kwargs) #retorna o modelo

# cds = convolve_and_downsample(8, 2, 4, 3, name='ngrama') #numero de amostras na entrada, dimensao do embedding, numero de filtros, tamanho do filtro
# print(cds.summary())

def cnn_embedding_softmax_model(vectorize_layer, vocab_size=vocab_size):
    input_layer = Input(shape=(1,), dtype=tf.string)
    x = input_layer
    x = vectorize_layer(x)
    x = Embedding(vocab_size, 2, name='projecao')(x)
    x = convolve_and_downsample(256, 2, 16, 4, name='ngramas')(x) #transoforma as palavras em N gramas, N = 3
    x = GlobalAveragePooling1D()(x)
    x = Dense(2, name='classificador')(x)
    x = Activation('softmax')(x)
    return Model(input_layer, x)


def blstm_softmax_model(vectorize_layer, vocab_size=vocab_size): #bi-directinal LSTM
    input_layer = Input(shape=(1,), dtype=tf.string)
    x = input_layer
    x = vectorize_layer(x)
    x = Embedding(vocab_size, 2, name='projecao')(x)
    x1 = LSTM(1)(x) #LSTM original
    x2 = LSTM(1, go_backwards=True)(x) #LSTM reversa, caminha na direcao oposta da camada original
    x = Concatenate()([x1, x2]) #concatenando as duas juntas 
    x = Dense(2, name='classificador')(x)
    x = Activation('softmax')(x)
    return Model(input_layer, x)


def avg_embedding_softmax_model(vectorize_layer, vocab_size=vocab_size):
    input_layer = Input(shape=(1,), dtype=tf.string)
    x = input_layer
    x = vectorize_layer(x)
    x = Embedding(vocab_size, 2, name='projecao')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(2, name='classificador')(x)
    x = Activation('softmax')(x)
    return Model(input_layer, x)
