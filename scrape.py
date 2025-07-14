import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
# ^^^imports^^^

# Get the HTML from the page
url = "https://fbref.com/en/comps/9/Premier-League-Stats"
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# Pulling multiple years of data
years = [2025, 2024, 2023, 2022, 2021]
all = []

for year in years:
    html = requests.get(url, headers=header)

    # Get the table from the HTML
    soup = BeautifulSoup(html.text, "lxml")
    selected = soup.select("table.stats_table")

    # if selected:
    #     table = selected[0]
    # else:
    #     print("\n\nCONTINUING\n\n")
    #     continue

    table = selected[0]

    # Get the link for each team's website
    links = [l.get("href") for l in table.find_all('a')]
    teams = [l for l in links if "/squads/" in l]
    team_links = [f"https://fbref.com{t}" for t in teams]

    # Going to the URL of the previous season
    prev_url = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com/{prev_url}"

    # Loop through each team:
    for team in team_links:
        name = team.split("/")[-1].replace("-Stats", "")

        # Extracting match data
        data = requests.get(team, headers=header)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

        # Extracting shooting data
        soup = BeautifulSoup(data.text, "lxml")
        links = [l.get("href") for l in soup.find_all('a')]
        shot_stats = [l for l in links if l and "all_comps/shooting/" in l]
        if not shot_stats:
            print("\n\nEMPTY\n\n")
        data = requests.get(f"https://fbref.com{shot_stats[0]}")
        shoot = pd.read_html(data.text, match="Shooting")[0]

        # Merging game and shooting data
        shoot.columns = shoot.columns.droplevel()
        try:
            team_data = matches.merge(shoot[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt", "npxG"]], on="Date")
        except:
            continue

        # Filter non-Prem competitions
        team_data = team_data[team_data["Comp"] == "Premier League"]

        # Add to the list of dataframes
        team_data["Season"] = year
        team_data["Team"] = name
        all.append(team_data)

        time.sleep(5)

# Combining dataframes
df = pd.concat(all)
df.to_csv("matches_5.csv")