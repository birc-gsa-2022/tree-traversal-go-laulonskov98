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

spy = yf.Ticker('SPY').history(start=pd.to_datetime("2006-01-01"))

df = pd.read_csv('nasdaq_data.csv', index_col=0)
start = 1000
length = 3030
benchmarkSpy = spy['Close'][start:length]
ax = benchmarkSpy.plot()

df = df[-length:-start]
dalist = df.index.tolist()
dalist.reverse()
yee = df.columns
yee = yee.tolist()
macro = pd.read_csv("C:/Users/Laurits/PycharmProjects/5factorredo/tdd-remake/FullMacroData.csv", index_col=0)




def run(lookback):
    datalistframe = pd.DataFrame(columns=macro.columns)
    labels = []

    startmoney = benchmarkSpy.iloc[0]
    startmulz = startmoney
    startmoneylist = [startmoney,startmoney]

    startmoney1 = benchmarkSpy.iloc[0]
    startmulz1 = startmoney1
    startmoneylist1 = [startmoney1,startmoney1]

    startmoney2 = benchmarkSpy.iloc[0]
    startmulz2 = startmoney2
    startmoneylist2 = [startmoney2,startmoney2]

    startmoney3 = benchmarkSpy.iloc[0]
    startmulz3 = startmoney3
    startmoneylist3 = [startmoney3,startmoney3]

    startmoneyshort = benchmarkSpy.iloc[0]
    startmulzshort = startmoneyshort
    startmoneylistshort = [startmoneyshort, startmoneyshort]

    startmoney1short = benchmarkSpy.iloc[0]
    startmulz1short = startmoney1short
    startmoneylist1short = [startmoney1short, startmoney1short]

    startmoney2short = benchmarkSpy.iloc[0]
    startmulz2short = startmoney2short
    startmoneylist2short = [startmoney2short, startmoney2short]

    startmoney3short = benchmarkSpy.iloc[0]
    startmulz3short = startmoney3short
    startmoneylist3short = [startmoney3short, startmoney3short]

    real = benchmarkSpy.iloc[0]
    realmulz = real
    reallist = []


    smamap = {}

    #lookback = 15
    window = 9  #for SMA of mean strategy

    for yunga in range(0,lookback+2):
        reallist.append(realmulz)


    for stock in yee:
        sma = pd.Series(df[stock]).rolling(window=window).mean().iloc[window - 1:]
        smamap[stock] = sma


    for index in range(1,length-start):
        date = dalist[index]
        row_current = df.iloc[-(index+1)]
        row_prev = df.iloc[-index]
        #print(row_current.name, row_prev.name)

        if index != 1:

            #print(row_current[active1], row_prev[active1])
            now = startmoney*(row_current[active1]/row_prev[active1])/2 + startmoney*(row_current[active12]/row_prev[active12])/2
            try:
                macc = macro.loc[date]
                if now > startmoney:

                    labels.append(1)
                else:
                    labels.append(-1)
                datalistframe = datalistframe.append(macc)
            except:
                pass
            startmoney = startmoney*(row_current[active1]/row_prev[active1])/2 + startmoney*(row_current[active12]/row_prev[active12])/2
            startmoneylist.append(startmoney)

            startmoney1 = startmoney1 * (row_current[active2] / row_prev[active2])/2 + startmoney1 * (row_current[active22] / row_prev[active22])/2
            startmoneylist1.append(startmoney1)

            startmoney2 = startmoney2 * (row_current[activemeanover1] / row_prev[activemeanover1])
            startmoneylist2.append(startmoney2)

            startmoney3 = startmoney3 * (row_current[activemeanunder1] / row_prev[activemeanunder1])/ 2 + startmoney3 * (
                          row_current[activemeanunder2] / row_prev[activemeanunder2]) / 2
            startmoneylist3.append(startmoney3)



            #startmoneyshort = startmoneyshort * (2 - row_current[active1] / row_prev[active1]) / 2 + \
            #                  startmoneyshort * (2 - row_current[active12] / row_prev[active12]) / 2
            #startmoneylistshort.append(startmoneyshort)

            #startmoney1short = startmoney1short * (2-row_current[active2] / row_prev[active2]) / 2 + \
            #                   startmoney1short * (2- row_current[active22] / row_prev[active22]) / 2
            #startmoneylist1short.append(startmoney1short)

            #startmoney2short = startmoney2short * (2 - row_current[activemeanover1] / row_prev[activemeanover1])
            #startmoneylist2short.append(startmoney2short)

            #startmoney3short = startmoney3short * (2-row_current[activemeanunder1] / row_prev[activemeanunder1]) / 2 + \
            #                   startmoney3short * (2-row_current[activemeanunder2] / row_prev[activemeanunder2]) / 2
            #startmoneylist3short.append(startmoney3short)


            if index > lookback:
                if index != lookback+1:
                    if best[0]=="1":
                        print(active1, active12)
                        real = real * (row_current[active1]/row_prev[active1])/2 + \
                               real*(row_current[active12]/row_prev[active12])/2
                        reallist.append(real)
                    if best[0]=="2":
                        print(active2, active22)
                        real = real * (row_current[active2] / row_prev[active2])/2 +\
                               real * (row_current[active22] / row_prev[active22])/2
                        reallist.append(real)
                    if best[0]=="3":
                        print(activemeanover1, activemeanover2)

                        real = real * (row_current[activemeanover1] / row_prev[activemeanover1])  #/2 + real * (
                                #row_current[activemeanover2] / row_prev[activemeanover2]) / 2
                        reallist.append(real)
                    if best[0]=="4":
                        print(activemeanunder1, activemeanunder2)
                        real = real * (row_current[activemeanunder1] / row_prev[activemeanunder1])/ 2 + \
                               real * (row_current[activemeanunder2] / row_prev[activemeanunder2]) / 2
                        reallist.append(real)

                    #if best[0]=="5":
                    #    real = real * (2-row_current[active1]/row_prev[active1])/2 + real*(2-row_current[active12]/row_prev[active12])/2
                    #    reallist.append(real)
                    #if best[0]=="6":
                    #    real = real * (2-row_current[active2] / row_prev[active2])/2 + real * (2-row_current[active22] / row_prev[active22])/2
                    #    reallist.append(real)
                    #if best[0]=="7":
                    #    real = real * (2-row_current[activemeanover1] / row_prev[activemeanover1])  #/2 + real * (
                    #            #row_current[activemeanover2] / row_prev[activemeanover2]) / 2
                    #    reallist.append(real)
                    #if best[0]=="8":
                    #    real = real * (2-row_current[activemeanunder1] / row_prev[activemeanunder1])/ 2 + real * \
                    #                  (2-row_current[activemeanunder2] / row_prev[activemeanunder2]) / 2
                    #    reallist.append(real)
                if index % lookback == 0 or index == lookback+1:

                    mon1change = startmoneylist[index]/startmoneylist[index-lookback]
                    mon2change = startmoneylist1[index] / startmoneylist1[index - lookback]
                    mon3change = startmoneylist2[index] / startmoneylist2[index - lookback]
                    mon4change = startmoneylist3[index] / startmoneylist3[index - lookback]
#
                    #mon5change = startmoneylistshort[index] / startmoneylistshort[index - lookback]
                    #mon6change = startmoneylist1short[index] / startmoneylist1short[index - lookback]
                    #mon7change = startmoneylist2short[index] / startmoneylist2short[index - lookback]
                    #mon8change = startmoneylist3short[index] / startmoneylist3short[index - lookback]
                    #, "5": mon5change,"6":mon6change, "7":mon7change,  "8":mon8change
                    veggies = {"1": mon1change,"2":mon2change, "3":mon3change,  "4":mon4change}
                    best = ("",0)
                    for key, value in veggies.items():
                        if value > best[1]:
                            best = (key, value)

        best1 = (1,0)
        best12 = (1,0)

        best2 = (2,0)
        best22 = (2, 0)

        bestmeanover1 = (0,0)
        bestmeanover2 = (0,0)

        bestmeanunder1 = (2,0)
        bestmeanunder2 = (2,0)
        i = 0
        for stock in yee:
            current = row_current.iloc[i]
            prev = row_prev.iloc[i]
            change = current/prev

            #bedste performer
            if change > best12[0]:
                if change > best1[0]:
                    best12 = best1
                    best1 = (change,i)
                else:
                    best12 = (change,i)

            #dårligste performer
            if change < best22[0]:
                if change < best2[0]:
                    best22 = best2
                    best2 = (change, i)
                else:
                    best22 = (change, i)

            if index > window:
                sma = smamap[stock]
                sma_current = sma.iloc[-index +window]
                sma_diff = current / sma_current

                # længst over mean (sma)
                if sma_diff > bestmeanover2[0]:
                    if sma_diff > bestmeanover1[0]:
                        bestmeanover2 = bestmeanover1
                        bestmeanover1 = (sma_diff, i)
                    else:
                        bestmeanover2 = (sma_diff, i)

                # længst under mean (sma)
                if sma_diff < bestmeanunder2[0]:
                    if sma_diff < bestmeanunder1[0]:
                        bestmeanunder2 = bestmeanunder1
                        bestmeanunder1 = (sma_diff, i)
                    else:
                        bestmeanunder2 = (sma_diff, i)

            i = i +1

        active1 = best1[1]
        active12 = best12[1]

        active2 = best2[1]
        active22 = best22[1]

        activemeanover1 = bestmeanover1[1]
        activemeanover2 = bestmeanover2[1]

        activemeanunder1 = bestmeanunder1[1]
        activemeanunder2 = bestmeanunder2[1]

    datamap = {"H_momentum": startmoneylist, "L_momentum":startmoneylist1, "H_mean":startmoneylist2, "L_mean":startmoneylist3}

    #hehedatamine = pd.DataFrame.from_dict(datamap)
    #algo = algos.UP()
    # run
    #result = algo.run(hehedatamine)
    #print(result.summary())
    #result.plot(weights=True, assets=True, ucrp=True, logy=True);

    startmoneydf = pd.DataFrame(startmoneylist, index=benchmarkSpy.index)
    sup = startmoneydf.plot(ax = ax)
    startmoneydf1 = pd.DataFrame(startmoneylist1, index=benchmarkSpy.index)
    sup2 = startmoneydf1.plot(ax = sup)
    startmoneydf2 = pd.DataFrame(startmoneylist2, index=benchmarkSpy.index)
    sup3 = startmoneydf2.plot(ax = sup2)
    startmoneydf3 = pd.DataFrame(startmoneylist3, index=benchmarkSpy.index)
    sup4 = startmoneydf3.plot(ax = sup3)

    startmoneydfreal = pd.DataFrame(reallist, index=benchmarkSpy.index)
    startmoneydfreal.plot(ax = sup4)

    def calc_sharpe(series):
        changes = series.pct_change()
        n = 255
        mean = changes.mean() * n - 0.01
        sigma = changes.std() * np.sqrt(n)
        sharpe = mean/sigma
        return sharpe

    sharpe1 = calc_sharpe(startmoneydf)
    sharpe2 = calc_sharpe(startmoneydf1)
    sharpe3 = calc_sharpe(startmoneydf2)
    sharpe4 = calc_sharpe(startmoneydf3)
    sharpe5 = calc_sharpe(startmoneydfreal)


    print("sharpes: ", sharpe4.values, sharpe3.values, sharpe2.values, sharpe1.values, sharpe5.values)


    bench = benchmarkSpy.iloc[-1]/benchmarkSpy.iloc[0]
    print("longs:")
    print(startmoney3/startmulz3, startmoney2/startmulz2, startmoney1/startmulz1, startmoney/startmulz, real/realmulz, bench)
    print("shorts:")
    print(startmoney3short/startmulz3short, startmoney2short/startmulz2short, startmoney1short/startmulz1short, startmoneyshort/startmulzshort)
    plt.show()

    return reallist, labels, datalistframe



resultasdf, labels, datalistframe = run(7)
datalistframe['labels'] = labels
datalistframe.to_csv("dailyTraining.csv")

def tester():
    lookbackMap = {}
    for x in range(4,11):
        resultasdf = run(x)
        lookbackMap[str(x)] = resultasdf
    hehedatamine = pd.DataFrame.from_dict(lookbackMap)
    indices_list_reversed = df.index.tolist()
    indices_list_reversed.reverse()
    hehedatamine.index = pd.Index(indices_list_reversed)

    algo = algos.UP(leverage=2)
    result = algo.run(hehedatamine)

    print(result.summary())
    result.plot(weights=True, assets=True, ucrp=True, logy=True)
    plt.show()

def trainForest():
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from datetime import timedelta

    startdate = "2022-02-01"
    enddate = "2022-02-26"

    data = pd.read_csv('C:/Users/Laurits/PycharmProjects/PredictiveOLPS/Main/dailyTraining.csv', index_col=0)
    macro = pd.read_csv('C:/Users/Laurits/PycharmProjects/5factorredo/tdd-remake/FullMacroData.csv', index_col=0)
    data = data.loc[:startdate]
    print(data)
    X = data.iloc[:, :len(data.columns) - 1]
    y = data.labels
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    startdate = pd.to_datetime(startdate)
    enddate = pd.to_datetime(enddate)

    while startdate < enddate:
        yeehaw = str(startdate)[0:10]
        try:
            macc = macro.loc[yeehaw]
        except KeyError:
            startdate += timedelta(1)
            continue
        maxx = np.array([macc.tolist()])
        prediction = rf.predict(maxx)
        proba = rf.predict_proba(maxx)
        print(yeehaw)
        print(prediction)
        print(proba)
        print("")
        startdate += timedelta(1)
trainForest()