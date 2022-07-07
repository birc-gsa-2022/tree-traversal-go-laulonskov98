import pandas as pd
import math
from datetime import date, timedelta

class momentum():

    def __init__(self, stocks, df):
        self.dailyReturns = []
        self.data = df
        self.stocks = stocks
        self.date = None

    def simulateDay(self, date, dailyprices, pricePrevious):
        self.date = date
        self.dailyReturns = []
        diffy = dailyprices / pricePrevious
        diffy = diffy.to_frame()
        diffy = diffy.dropna()
        diffy = diffy.sort_values(by=0, ascending=False)
        for index, value in diffy.iterrows():
            self.dailyReturns.append((index, value[0]))
