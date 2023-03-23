from dcToken import apiToken
import discord
from mal_api import command_decoder
import re
from crawler import crawl

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
    
    else: 

        terms = re.findall('\w+', message.content.lower())

        if terms:
            if (terms[0] == 'crawl'):
                await message.channel.send("crawling")
                crawl((message.content, 0))
                
            elif (terms[0] != 'run') or len(terms) <= 1:
                await message.channel.send(error_message)
        
            text = command_decoder(message.content.lower())

            if text:
                await message.channel.send("os animes da temporada sao:")
                for t in text[1]:
                    sleep(2)
                    await message.channel.send(t)

        #else: await message.channel.send("There was a issue with the command, please make sure you tiped everything correctly")



client.run(apiToken)