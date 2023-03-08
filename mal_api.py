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

    terms = re.findall('\w+', text)

    if terms[0] != 'run':
        return([0, "Incorrect command, please try !help to read available commands"])

    
    # match = re.search(r"!run anime: (\S+)", text)

    # if match:
    #     anime_name = match.group(1)
    #     return request_maker('anime', anime_name)


    match = re.search(r"!run year:\s*([a-zA-Z]+)\s*season:\s*(\d{4})", text, re.IGNORECASE)
    if match:
        season = match.group(1)
        year = match.group(2)
        data = (season, year)
        return request_maker('season', data)

    else:
        return([0, "There was a issue with the command, please make sure you tiped everything correctly"])




def request_maker(request, data):



    # if request == 'anime':
    #     anime_title = data
    
    #     # Make a request to the MyAnimeList API to search for the anime
    #     header = {
    #         "X-MAL-CLIENT-ID":  CLIENT_ID
    #     }
    #     params = {
    #         "q": anime_title
    #     }

    #     anime_search_response = requests.get("https://api.myanimelist.net/v2/anime", headers=header, params=params)

    #     # Get the ID of the first anime in the search results
    #     anime_data = anime_search_response.json()['data']
    #     if len(anime_data) == 0:
    #         print(f"No results found for anime: {anime_title}")
    #     else:
    #         anime_id = anime_data[0]["node"]["id"]

    #         # Use the ID to make a request for the anime information
    #         anime_response = requests.get(f"https://api.myanimelist.net/v2/anime/{anime_id}", headers=header)

    #         # Print the anime information
    #         anime_info = anime_response.json()
            
    #         print(f"Title: {anime_info['title']}")
    #         print(f"Synopsis: {anime_info['synopsis']}")
    #         print(f"Start Date: {anime_info['start_date']}")
    #         print(f"End Date: {anime_info['end_date']}")
    #         return anime_info





    if request == 'season':
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

        return [1, titles]


