import pandas as pd
import json
from bs4 import BeautifulSoup
import requests
import os
from datetime import datetime, date, time, timedelta
today = datetime.today()
mm = today.month
yy = today.year


dataPath = 'C:/Users/nhche/Development/Cryptoprices/Data/Cryptocomp/'

url = "https://api.coinmarketcap.com/v1/ticker/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
dic = json.loads(soup.prettify())

df = pd.DataFrame(columns = ["Ticker", "MarketCap"])

for i in range(len(dic)):
    df.loc[len(df)] = [dic[i]['symbol'] , dic[i]['market_cap_usd']]

df.sort_values(by=['MarketCap'])
df.MarketCap = pd.to_numeric(df.MarketCap)
fileName = "CryptoTickersByCap_"+str(mm) + "_" + str(yy) + ".txt"

p = df[df.MarketCap > 20e6]
p.to_csv(os.path.join(dataPath, fileName),",");

