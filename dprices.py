import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime, time
import json
from bs4 import BeautifulSoup
import requests
import os
from win32com.client import Dispatch
import time as mod_time

datapath = "C:/CryptoCNN/"
dataPath = os.path.join(datapath, "pricedatabase/")

if not os.path.exists(dataPath):
    os.mkdir(dataPath)

def t2d (timestamp):
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
def d2t (date):
    return datetime.strptime(date_today, '%Y-%m-%d').timestamp()
def fetchCryptoOHLC(fsym, tsym):
    cols = ['date','time','open','high','low','close','volume']
    lst = ['time', 'open', 'high', 'low', 'close','volumeto']

    timestamp_today = mod_time.mktime(datetime.today().timetuple()) + datetime.today().microsecond/1e6
    currtime = timestamp_today

    for j in range(2):
        df = pd.DataFrame(columns = cols)
        url = "https://min-api.cryptocompare.com/data/histoday?fsym=" + fsym + "&tsym=" + tsym + "&toTs=" + str(int(currtime)) + "&limit=2000"
        response = requests.get(url)

        soup = BeautifulSoup(response.content, "html.parser")
        dic = json.loads(soup.prettify())
        for i in range(1,2001):
            tmp = []
            for e in enumerate(lst):
                x = e[0]
                y = dic['Data'][i][e[1]]
                if(x == 0):
                    tmp.append(str(t2d(y)))
                tmp.append(y)
            if (np.sum(tmp[-4::]) > 0):
                df.loc[len(df)] = np.array(tmp)
        df.index = pd.to_datetime(df.date)
        df.drop('date', axis = 1, inplace = True)
        curr_timestamp = int(df.ix[0][0])
        if(j==0):
            df0 = df.copy()
        else:
            data = pd.concat([df,df0], axis = 0)
    return data

df = pd.DataFrame(columns=["Ticker", "MarketCap"])
url = "https://api.coinmarketcap.com/v1/ticker/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
dic = json.loads(soup.prettify())

for i in range(len(dic)):
    df.loc[len(df)] = [dic[i]['symbol'], dic[i]['market_cap_usd']]

df.sort_values(by=['MarketCap'])
df.MarketCap = pd.to_numeric(df.MarketCap)
p = df[df.MarketCap > 40e6]
p.to_csv(os.path.join(datapath, "CryptoTickersByCap.txt"), ",")
portfolio = list(p.Ticker)

tsym = 'USD'
print('getting price histories for high Cap currencies')
for e in enumerate(portfolio[1:10]):
    if e[1].startswith("MIOTA") or e[1].startswith("LKK"):
        continue
    data = fetchCryptoOHLC(e[1],tsym)
    print(data["2017-10-20"])
    fileend = e[1] + '_' + tsym + '.csv'
    print (fileend)
    data.to_csv(os.path.join(dataPath, fileend),',')