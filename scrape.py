import requests
from bs4 import BeautifulSoup
# ^^^imports^^^

# Get the HTML from the page
url = "https://fbref.com/en/comps/9/Premier-League-Stats"
html = requests.get(url)

# Get the table from the HTML
soup = BeautifulSoup(html.text)
table = soup.select("table.stats_table")[0]
links = table.find_all("a")

# Go to each team's website
for l in links:
    l.get("href")

print(l)