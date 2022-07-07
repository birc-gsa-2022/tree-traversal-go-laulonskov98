import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta
import numpy as np
import seaborn as sns
import universal as up
from universal import algos
from universal.algos import *
import sys
import math

sys.path.append("C:/Users/Laurits/PycharmProjects/PredictiveOLPS/Main/strategies")
import meanReverting
import momentum
import main
import matplotlib.pyplot as plt


def calc_sharpe(series):
    changes = series.pct_change()
    n = 255
    mean = changes.mean() * n - 0.01
    sigma = changes.std() * np.sqrt(n)
    sharpe = mean / sigma
    return sharpe

#generer labels for hver så hver strategi kan risikojusteres

#main.run()
factory = []
strategies = []
df = pd.read_csv('nasdaq_data.csv', index_col=0)
try:
    df.drop('MNMD', inplace=True, axis=1)
except KeyError:
    pass
try:
    df.drop('CLSK', inplace=True, axis=1)
except KeyError:
    pass

df = df.iloc[:1000]
print(df)
dalist = df.index.tolist()
dalist.reverse()
print(dalist)
stocks = df.columns
stocks = stocks.tolist()
print(stocks)

meanrevert = meanReverting.meanReverting(stocks, df)
momentum = momentum.momentum(stocks, df)
prev = dalist[8]
spytotal = total = 100

totalvalue = []
spyvalue = []
spy = yf.Ticker('SPY').history(start=pd.to_datetime("2006-01-01"))
isFirst = True
for date in dalist[9:-2]:   #last date doesnt work for some reason
    if not isFirst:
        # add spychange to plotlist
        spyprev = spy.loc[prev].Close
        spynow = spy.loc[date].Close
        spytotal *= spynow / spyprev
        spyvalue.append(spytotal)
        # add changes for every list of
        print(date," : ")
        print("momdate", momentum.date)
        print("Longs")
        for resultingList in longlists:
            for tick in resultingList:
                previous_price = df.loc[prev][tick[0]]
                today_price = df.loc[date][tick[0]]
                change = 2- today_price / previous_price
                weightchange = 1 + (change-1)/6
                if change == math.isnan(change):
                    continue
                print(tick, today_price, previous_price, change, weightchange)
                total *= weightchange
        print("SHORTS")
        for resultingList in shortlists:
            for tick in resultingList:
                previous_price = df.loc[prev][tick[0]]
                today_price = df.loc[date][tick[0]]
                change = today_price / previous_price
                if change == math.isnan(change):
                    continue

                weightchange = 1 + (change-1)/6
                print(tick, today_price, previous_price, change, weightchange)

                total *= weightchange
        print("total:", total)
        print("")
        totalvalue.append(total)
    else:
        isFirst = False
    prices = df.loc[date]
    prevpass = df.loc[prev]
    momentum.simulateDay(date, prices, prevpass)
    meanrevert.simulateDay(date, prices)

    top2_mom = momentum.dailyReturns[0:2]
    bot1_mom = momentum.dailyReturns[-1:]
    top2_mean = meanrevert.dailyReturns[0:1]
    bot1_mean = meanrevert.dailyReturns[-1:]

    longlists = [top2_mom, top2_mean]
    shortlists = [bot1_mom, bot1_mean]
    prev = date

startmoneydf = pd.DataFrame(totalvalue, index=df.index[10:-2])
sharpe = calc_sharpe(startmoneydf)
print("sharpe: ", sharpe)

print(spytotal, total)
plt.plot(dalist[10:-2], spyvalue)
plt.plot(dalist[10:-2], totalvalue)
plt.show()

