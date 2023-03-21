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

def crawl(command):

    #tratando a url:
    match = re.match(r"!crawl (.+)", command)

    if match:
        url = match.group(1)
    
    #recebendo a url e fazendo o request:
    web_page = requests.get(url)
    web_page.encoding = 'utf-8'

    #usando o beufifull soup para tratar a pagina
    soup = BeautifulSoup(web_page.content, 'html.parser')

    #extraindo o titulo e o body da pagina
    title = soup.title.string.strip()
    body = soup.get_text().strip()


    #salvando o conteudo em um csv:
    
    # Create a DataFrame with the data
    df = pd.DataFrame({'url': [url], 'title': [title], 'body': [body]})
    
    # Save the DataFrame to a CSV file
    with open('crawler_data.csv', mode='a', newline='') as file:
        df.to_csv(file, header=not file.tell(), index=False)

