import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
# ^^^imports^^^

# Get the HTML from the page
url = "https://fbref.com/en/comps/9/Premier-League-Stats"
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# Pulling multiple years of data
years = [2025, 2024, 2023]
all = []

for year in years:
    print("year:", year)
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

    # Going to the URL of the previous season
    prev_url = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com/{prev_url}"


    # Loop through each team:
    for team in team_links:
        print("team:", team)
        name = team.split("/")[-1].replace("-Stats", "")
        print("name:", name)

        # Extracting match data
        data = requests.get(team, headers=header)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

        # Extracting shooting data
        soup = BeautifulSoup(data.text, "lxml")
        links = soup.find_all("a")
        print("links:", links)
        shooting = []
        for l in links:
            shooting.append(l.get("href"))
        shot_stats = []
        print("shooting:", shooting)
        for s in shooting:
            if s and "all_comps/shooting/" in s:
                shot_stats.append(s)
        print("shots stats:", shot_stats)
        if not shot_stats:
            print("\n\nEMPTY\n\n")
        data = requests.get(f"https://fbref.com{shot_stats[0]}")
        shoot = pd.read_html(data.text, match="Shooting")[0]

        # Merging game and shooting data
        shoot.columns = shoot.columns.droplevel()
        try:
            team_data = matches[0].merge(shoot[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
            print("finish try")
        except:
            print("caught")
            continue

        # Filter non-Prem competitions
        team_data = team_data[team_data["Comp"] == "Premier League"]

        # Add to the list of dataframes
        team_data["Season"] = year
        team_data["Team"] = team
        all.append(team_data)

        time.sleep(10)

# Combining dataframes
df = pd.concat(all)
df.to_csv("matches.csv")