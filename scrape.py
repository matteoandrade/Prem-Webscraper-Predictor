import requests
import pandas as pd
from bs4 import BeautifulSoup
# ^^^imports^^^

# Get the HTML from the page
url = "https://fbref.com/en/comps/9/Premier-League-Stats"


# Pulling multiple years of data
years = [2025, 2024, 2023]
all = []

for year in years:
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

    # Loop through each team:
    for team in team_links:
        name = team.split("/")[-1].replace("-Stats", "")

        # Extracting match data
        data = requests.get(team)
        matches = pd.read_html(data.text, match = "Scores & Fixtures")[0]

        # Extracting shooting data
        soup = BeautifulSoup(data.text, "lxml")
        links = soup.find_all("a")
        shooting = []
        for l in links:
            shooting.append(l.get("href"))
        shot_stats = []
        for s in shooting:
            if s and "all_comps/shooting/" in s:
                shot_stats.append(s)
        data = requests.get(f"https://fbref.com{shot_stats[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]

        # Merging game and shooting data
        shooting.columns = shooting.columns.droplevel()
        try:
            team = matches[0].merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except:
            continue