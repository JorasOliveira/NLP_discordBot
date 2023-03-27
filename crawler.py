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
#TODO - tirar os prints dps de td estiver pronto

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
