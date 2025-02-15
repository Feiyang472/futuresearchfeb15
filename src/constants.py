import requests
from functools import cache

@cache
def all_available_indexes():
    url = "https://index.commoncrawl.org/collinfo.json"
    response = requests.get(url)
    return response.json()
