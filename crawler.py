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

def crawl(command, files = 0, max_files = 5):

    # if files >= max_files:
    #     return

    print(command[0])
    print(command[1])

    if command[1] == 0:
        #tratando a url:
        match = re.match(r"!crawl (.+)", command[0])

        if match:
            url = match.group(1)
        
        web_page = requests.get(url)
        web_page.encoding = 'utf-8'

    if command[1] == 1:
        #recebendo a url e fazendo o request:
        web_page = requests.get(command[0])
        web_page.encoding = 'utf-8'



    #usando o beufifull soup para tratar a pagina
    soup = BeautifulSoup(web_page.content, 'html.parser')

    # Get the title and body of the page
    try:
        title = soup.title.string.strip()
    except AttributeError:
        return
    try:
        body = soup.get_text().strip()
    except AttributeError:
        return

   
    #exrtaindo os conteudos dos links recursivamente:

    print(files)

    if files >= max_files:
        return

    else:
#TODO - o chatbot ta retornando logo no primeiro link, tem que fazer ele salvar e continuar crawling ate 
# salvar o numero de arquivos requisitado
        #extraindo os links:
        links = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href').startswith('https://')]

        for link in links:
            # Check if the URL already exists in the CSV file

            if os.path.isfile(f'crawler_data.csv'):
                existing_df = pd.read_csv('crawler_data.csv')
                if link in existing_df['url'].values:
                    return

            else:

                #salvando o conteudo em um csv:
                # Create a DataFrame with the data
                df = pd.DataFrame({'url': [url], 'title': [title], 'body': [body]})
            
                # Save the DataFrame to a CSV file
                with open('crawler_data.csv', mode='a', newline='') as file:
                    df.to_csv(file, header=not file.tell(), index=False)
                    files += 1


                crawl((link,1), files, max_files = 5)



