import requests
import pandas as pd
from bs4 import BeautifulSoup
# ^^^imports^^^

# Get the HTML from the page
url = "https://fbref.com/en/comps/9/Premier-League-Stats"
html = requests.get(url)

# Get the table from the HTML
soup = BeautifulSoup(html.text, "lxml")
table = soup.select("table.stats_table")[0]
links = table.find_all("a")

# Get the link for each team's website
link_lst = []
for l in links:
    link_lst.append(l.get("href"))
teams = []
for l in link_lst:
    if "/squads/" in l:
        teams.append(l)
team_links = [f"https://fbref.com{t}" for t in teams]
    
# Extracting match data
url = team_links[0]
data = requests.get(url)
matches = pd.read_html(data.text, match = "Scores & Fixtures")

# Extracting shooting data
soup = BeautifulSoup(data.text)
links = soup.find_all("a")
shooting = []
for l in links:
    shooting.append(l.get("href"))
shot_stats = []
for s in shooting:
    if s and "all_comps/shooting/" in s:
        shot_stats.append(s)

data = requests.get(f"https://fbref.com{links[0]}")
shooting = pd.read_html(data.text, match="Shooting")[0]