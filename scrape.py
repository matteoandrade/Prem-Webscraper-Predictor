import requests
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
    
print("\n", team_links)