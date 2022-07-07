import pandas as pd
import math

class meanReverting():

    def __init__(self, stocks, df):
        self.dailyReturns = []
        self.smaData = {}
        self.stocks = stocks
        df = df[::-1]
        for stock in stocks:
            sma = pd.Series(df[stock]).rolling(window=9).mean().iloc[9 - 1:]
            sma = sma[::-1]
            self.smaData[stock] = sma

    def simulateDay(self, date, dailyprices):
        self.dailyReturns = []
        for stock in self.stocks:
            sma = self.smaData[stock][date]
            price = dailyprices[stock]
            if math.isnan(price/sma):
                continue
            self.dailyReturns.append((stock, price/sma))
        self.dailyReturns = sorted(self.dailyReturns, key=lambda tup: tup[1], reverse=True)
