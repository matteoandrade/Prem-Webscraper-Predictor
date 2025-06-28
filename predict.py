import pandas as pd
# ^^^ imports ^^^

matches = pd.read_csv("matches.csv", index_col=0)

print(matches.head())
print(matches.shape)