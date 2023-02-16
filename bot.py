import dcToken
import discord

#print(dcToken.apiToken)

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



client.run(dcToken.apiToken)