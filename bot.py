from dcToken import apiToken
import discord
from mal_api import command_decoder
import re
from time import sleep

#client = discord.Client()

intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    guild = discord.utils.get(client.guilds, name='A Cidade dos Robôs')
    channel = discord.utils.get(guild.text_channels, name='bot-fest')
    await channel.send('O bot está online!')




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
            await message.channel.send("Use '!run season: XXXX, year: NNNN' para saber os animes que lancaram na temporada e no ano expecificado, as temporadas sao divididas em: FALL, WINTER, SUMMER, SPRING. e use 4 digitos para o ano.")
            await message.channel.send('Todos os dados sao extraidos do My Anime List, https://myanimelist.net/')
    
    else: 
        text = command_decoder(message.content.lower())
       

        if text[0] == 1:
            await message.channel.send("os animes da temporada sao:")
            for t in text[1]:
                sleep(2)
                await message.channel.send(t)
        else: await message.channel.send(text[1])



client.run(apiToken)