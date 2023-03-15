import requests
from dcToken import malID
import re
import json
import secrets

# Your MyAnimeList API credentials
CLIENT_ID = malID
CLIENT_SECRET = ' '

# The year and season you want to search for (e.g. "2021", "summer")
year = "2021"
season = "summer"

def command_decoder(text):


    print(text)
    
    terms = re.findall('\w+', text)

    if terms[0] != 'run':
        return(0)

    match = re.search(r"!run season:\s*([a-zA-Z]+)\s*,\s*year:\s*(\d{4})", text, re.IGNORECASE)
    print(match)
    if match:
        print(1)
        season = match.group(1)
        year = match.group(2)
        data = (season, year)
        
        return request_maker('season', data)

    else:
        return(False)




def request_maker(request, data):

    print(2)
    if request == 'season':
        print(3)
        # Make a request to the MyAnimeList API to get the access token
        headers = {
            "X-MAL-CLIENT-ID":  CLIENT_ID
        }
        season_year =  data[1]
        season = data[0]
        sort =  "anime_score"
        limit = 10
        #list.net/v2/anime/season/2017/summer?limit=4' 
        request_url = "https://api.myanimelist.net/v2/anime/season/%s/%s?%d/sort=anime_score" %(season_year, season, limit)
        anime_response = requests.get(request_url, headers=headers)

        titles = []

        # Print the anime titles
        anime_data = anime_response.json()
        data = anime_data['data']
        for i in range(len(anime_data['data'])):
            titles.append(anime_data['data'][i]['node']['title'])

        return [True, titles]


