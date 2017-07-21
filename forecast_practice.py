import pandas as pd
from matplotlib import pyplot as plt

import datetime as DT
import matplotlib.dates as mdates

from random import gauss
from random import seed
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import calendar

##df = pd.read_csv('monthly-beer-production-in-austr.csv',header=None,
##                 skiprows=1,skipfooter=2,engine='python',
##                 names=['date','beer'])
##df['date'] = df['date'].apply(lambda x:dt.datetime.strptime(x,"%Y-%m"))
##df.set_index('date', inplace=True)
##df.plot()

##Rcode-------------------------
##a10.rda converted to csv file in R console as follows:
##setwd('D:/hyndman_forecasting')
##load('a10.rda')
##ls()
##a10
##write.zoo(a10,"a10.csv",index.name="Date",sep=",")
##------------------------------

def t2dt(atime):
    year = int(atime)
    remainder = atime - year
    boy = DT.datetime(year, 1, 1)
    eoy = DT.datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + DT.timedelta(seconds=seconds)

##melsydPlot----------------------------------
def melsyd_plot():
    df3 = pd.read_csv('./data/melsyd.csv')
    df3['Date'] = df3['Date'].apply(lambda x:t2dt(x))
    df3['Date'] = df3['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df3.set_index('Date', inplace=True)
    #print(df3.tail())
    fig,ax = plt.subplots()
    ax.plot_date(df3.index,df3['Economy.Class'],'-')
    plt.title('Economy class passengers: Melbourne-Sydney')
    plt.ylabel('Thousands')
    plt.xlabel('Year')
    plt.show()

##a10Plot-------------------------------------
def a10_plot():
    df2 = pd.read_csv('a10.csv',header=None,names=['date','drug_sales'])
    df2['date'] = df2['date'].apply(lambda x:t2dt(x))
    #df2['date'] = df2t['date'].apply(lambda x: x.strftime('%Y-%m'))
    df2.set_index('date', inplace=True)
    #print(df2.head(10))
    fig,ax = plt.subplots()
    ax.plot_date(df2.index,df2['drug_sales'],'-')
    plt.title('Antidiabetic drug sales')
    plt.ylabel('$ million')
    plt.show()

##SeasonPlot----------------------------------
def season_plot():
    robjects.r['load']("./data/a10.rda")
    #print(robjects.r['a10'])
    pandas2ri.activate()
    pydf = pd.DataFrame(robjects.r['a10'])
    #print(pydf.head())
    dtrng = pd.date_range("1991-07","2008-06",freq='MS')
    pydf.set_index(dtrng, inplace=True)
    pydf = pydf.rename(columns={0:'drug_sales'})
    #print(pydf.tail())
    #pydf.plot()
    #plt.title('Antidiabetic drug sales')
    #plt.ylabel('$ million')
    #plt.show()
    pv = pd.pivot_table(pydf,
                    index=pydf.index.month,
                    columns=pydf.index.year,
                    values='drug_sales',
                    aggfunc='sum')
    fig,ax = plt.subplots()
    pv.plot(ax=ax,style='o-',legend=None)
    xtks = ax.get_xticks()[1:]
    ax.set_xticks(xtks)
    xlabels = [calendar.month_abbr[int(x)] for x in xtks.tolist()]
    ax.set_xticklabels(xlabels)
    plot_margin = 1
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin,x1 + plot_margin,y0,y1))
    plt.show()

##MonthPlot----------------------------------


##WhiteNoise---------------------------------
def plot_noise():
    # seed random number generator
    seed(30)
    # create white noise series
    series = [gauss(0.0, 1.0) for i in range(50)]
    series = Series(series)
    # summary stats
    print(series.describe())
    # prelims for subplots
    fig,ax = plt.subplots(nrows=2, ncols=2)
    # line plot
    series.plot(ax=ax[0,0])
    ax[0,0].set_title('White Noise')
    # histogram plot
    series.hist(ax=ax[0,1])
    ax[0,1].set_title('Noise Histogram')
    # autocorrelation
    autocorrelation_plot(series,ax=ax[1,0])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #melsyd_plot()
    #a10_plot()
    season_plot()
    #plot_noise()

