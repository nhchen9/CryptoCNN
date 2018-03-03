from math import pi
import pandas as pd
import bokeh
import numpy as np
import numpy.random as npr
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.stocks import MSFT
import bokeh.io

def genBoundTri(mean, length, delta, vol):
    path = "C:/CryptoCNN/pricedatabase/ETH_USD.csv"
    df = pd.read_csv(path, header = 0, sep =',',parse_dates = True)[:length]
    low = mean - delta
    high = mean + delta
    df.open[0] = low + delta * npr.random()
    upward = -1
    if (npr.random() > .5):
        upward = 1
    df.close[0] = npr.random() * upward *vol  * df.open[0] + df.open[0]
    for i in range(1,length):
        low = mean - delta + i/length * delta
        high = mean + delta - i/length*delta
        if(df.close[i-1] < low):
            upward = upward * -1
        df.open[i] = df.close[i-1]
        df.close[i]  = npr.random() * upward * vol * (1-i/length)  * df.open[i] + df.open[0]
    return df
def writeToFile(df, number):
    df["date"]= pd.to_datetime(df["date"])
    inc = df.close > df.open
    dec = df.open > df.close
    w = 12*60*60*1000 # half day in ms
    p = figure(x_axis_type="datetime", plot_width=1000, title = "ETH Candlestick")
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha=0.3
    p.segment(df.date, df.high, df.date, df.low, color="black")
    p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
    p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")
    fname = str(number) + ".png"    
    bokeh.io.export_png(p, filename=fname)

mean = 100
for i in range (1,3):
    for j in range(1,3):
        for k in range(1,5):
            df = genBoundTri(mean, 50 * i, .1 *j* mean, .3 + .05*k)
            writeToFile(df, 25*i + 5*j + k)
            print(25*i+5*j+k)

''' winners - (1,1,4), (2,2,4) ''' 
