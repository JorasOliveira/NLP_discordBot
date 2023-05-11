from dcToken import apiToken
import discord
from mal_api import command_decoder
import re
from crawler import crawl
from search import tfidf_search
from search import wn_search
from search import train

#client = discord.Client()

intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)


error_message = "Incorrect command, please try !help to read available commands"

@client.event
async def on_ready():
    guild = discord.utils.get(client.guilds, name='A Cidade dos Robôs')
    channel = discord.utils.get(guild.text_channels, name='bot-fest')
    #await channel.send('O bot está online!')




@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower() == '!oi':
        await message.channel.send('Olá!')

    if message.content.lower() == '!source':
            await message.channel.send('Ola, meu codigo fonte esta em: https://github.com/JorasOliveira/NLP_discordBot')

    if message.content.lower() == '!author':
            await message.channel.send('Meu autor eh o Jorás, o email dele eh: jorascco@al.insper.edu.br')

    if message.content.lower() == '!help':
            await message.channel.send("Use '!run season: XXXX,year: NNNN' para saber os animes que lancaram na temporada e no ano expecificado, as temporadas sao divididas em: FALL, WINTER, SUMMER, SPRING. e use 4 digitos para o ano.")
            await message.channel.send('Todos os dados sao extraidos do My Anime List, https://myanimelist.net/')
            await message.channel.send("Use '!source para saber aonde esta o meu codigo fonte." )
            await message.channel.send("Use '!auhtor para saber o nome e email do meu autor")
            await message.channel.send("Use '!crawl para eu apredner novos dados, isto pode demorar alguns minutos")
            await message.channel.send("Use '!train para eu re-treinar com os novos dados, isto pode demorar alguns minutos")
            await message.channel.send("Use '!search' para eu procurar um termo na minha base de dados")  
            await message.channel.send("Use '!wn_search' para eu procurar um termo na minha base de dados, usando a wordnet")  
    else:
        terms = re.findall('\w+', message.content.lower())

        if terms:
            if (terms[0] == 'crawl'):
                await message.channel.send("crawling")
                crawl((message.content, 0))
                await message.channel.send("Recomendo rodar o comando !train para me re-treinar com os novos dados")

            elif (terms[0] == 'search'):
                await message.channel.send("searching")
                result =  tfidf_search(message.content)
                await message.channel.send(result)

            elif (terms[0] == 'wn_search'):
                await message.channel.send("searching with wordnet")
                result =  wn_search(message.content)
                await message.channel.send(result)
            
            elif (terms[0] == 'train'):
                await message.channel.send("aprendendo!, infelizmente nao vou conseguir conversar com voce ate eu terminar de aprender, isto deve demorar poucos minutos, agradeco a paciencia!")
                train()
                await message.channel.send("aprendi!, posso voltar a conversar")

            elif (terms[0] != 'run') or (terms[0] != 'author') or (terms[0] != 'source') or len(terms) < 1:
                await message.channel.send(error_message)
        
            text = command_decoder(message.content.lower())

            if text:
                await message.channel.send("os animes da temporada sao:")
                for t in text[1]:
                    await message.channel.send(t)




client.run(apiToken)