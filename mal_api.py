import requests
import dcToken
import re
import json
import secrets

# Your MyAnimeList API credentials
CLIENT_ID = 'a431c07618a9813edc297baba3bddd59' 
CLIENT_SECRET = "1f6bdb59a20e6f5ac37b5b26ac1cc23c7f2563d539825259728be7f3e7c6747b"

# The year and season you want to search for (e.g. "2021", "summer")
year = "2021"
season = "summer"

def command_decoder(text):

    terms = re.findall('\w+', text)

    if terms[0] != 'run':
        return("Incorrect command, please try !help to read available commands")

    
    match = re.search(r"!run anime: (\S+)", text)

    if match:
        anime_name = match.group(1)
        print(anime_name)
        return request_maker('anime', anime_name)


    match = re.search(r"!run season: (\S+) year: (\d{4})", text)
    if match:
        season = match.group(1)
        year = match.group(2)
        data = (season, year)
        print(f"Season: {season}, Year: {year}")
        return request_maker('season', data)

    else:
        return("There was a issue with the command, please make sure you tiped everything correctly")




def request_maker(request, data):



    if request == 'anime':
        anime_title = data
    
        # Make a request to the MyAnimeList API to search for the anime
        header = {
            "X-MAL-CLIENT-ID":  CLIENT_ID
        }
        params = {
            "q": anime_title
        }

        anime_search_response = requests.get("https://api.myanimelist.net/v2/anime", headers=header, params=params)

        # Get the ID of the first anime in the search results
        anime_data = anime_search_response.json()['data']
        if len(anime_data) == 0:
            print(f"No results found for anime: {anime_title}")
        else:
            anime_id = anime_data[0]["node"]["id"]

            # Use the ID to make a request for the anime information
            anime_response = requests.get(f"https://api.myanimelist.net/v2/anime/{anime_id}", headers=header)

            # Print the anime information
            anime_info = anime_response.json()
            
            print(f"Title: {anime_info['title']}")
            print(f"Synopsis: {anime_info['synopsis']}")
            print(f"Start Date: {anime_info['start_date']}")
            print(f"End Date: {anime_info['end_date']}")
            return anime_info





    if request == 'season':
        # Make a request to the MyAnimeList API to get the access token
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        # auth_data = {
        #     'X-MAL-CLIENT-ID:': CLIENT_ID,
        # }

        auth_response = requests.post("https://api.myanimelist.net/v2/oauth2/token", data=auth_data)
        access_token = auth_response.json()["access_token"]

        # Use the access token to make a request for the anime information
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        params = {
            "season_year": data[1],
            "season": data[0],
            "sort": "anime_score",
            "limit": 10
        }
        anime_response = requests.get("https://api.myanimelist.net/v2/anime", headers=headers, params=params)

        # Print the anime titles
        anime_data = anime_response.json()["data"]
        return anime_data
        # for anime in anime_data:
        #     print(anime["title"])

