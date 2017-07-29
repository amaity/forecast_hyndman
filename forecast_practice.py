import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import datetime as dt
import matplotlib.dates as mdates

from random import gauss
from random import seed
from pandas import Series

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import calendar
import itertools

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
##TimeUtil------------------------------
def t2dt(atime):
    """
    Convert atime (a float) to dt.datetime
    This is the inverse of dt2t.
    assert dt2t(t2dt(atime)) == atime
    """
    year = int(atime)
    remainder = atime - year
    boy = dt.datetime(year, 1, 1)
    eoy = dt.datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + dt.timedelta(seconds=seconds)

def dt2t(adatetime):
    """
    Convert adatetime into a float. The integer part of the float should
    represent the year.
    Order should be preserved. If adate<bdate, then d2t(adate)<d2t(bdate)
    time distances should be preserved: If bdate-adate=ddate-cdate then
    dt2t(bdate)-dt2t(adate) = dt2t(ddate)-dt2t(cdate)
    """
    year = adatetime.year
    boy = dt.datetime(year, 1, 1)
    eoy = dt.datetime(year + 1, 1, 1)
    return year + ((adatetime - boy).total_seconds() / \
                   ((eoy - boy).total_seconds()))

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
    df2 = pd.read_csv('./data/a10.csv',header=None,names=['date','drug_sales'])
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
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.api as sm 
def month_plot():
    df = pd.read_csv('./data/a10.csv',header=None,names=['date','drug_sales'])
    dtrng = pd.date_range("1991-07","2008-06",freq='MS')
    df.set_index(dtrng, inplace=True)
    df.drop('date', axis=1, inplace=True)
    fig, ax = plt.subplots()
    sm.graphics.tsa.month_plot(df,ylabel='$ million',ax=ax)
    ax.set_title('Seasonal deviation plot: antidiabetic drug sales')
    #gp = df.groupby([df.index.month,df.index.year]).sum()
    #tsaplots.seasonal_plot(gp,list(range(1,13)))
    plt.show()

##ScatterPlot--------------------------------
def scatter_plot():
    df = pd.read_csv('./data/fuel.csv')
    df['City'] = np.random.normal(df['City'], 0.04)
    df['Carbon'] = np.random.normal(df['Carbon'], 0.04)
    df.plot.scatter(x='City',y='Carbon')
    plt.show()

##PairsPlot----------------------------------
##Ref:https://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
def plot_scatter_matrix():
##    np.random.seed(1977)
##    numvars, numdata = 4, 10
##    data = 10 * np.random.random((numvars, numdata))
##    fig = scatterplot_matrix(data, ['mpg', 'disp', 'drat', 'wt'],
##            linestyle='none', marker='o', color='black', mfc='none')
    df = pd.read_csv('./data/fuel.csv')
    data = df[['Litres','City','Highway','Carbon']].values.T
    fig = scatterplot_matrix(data, ['Litres','City','Highway','Carbon'],
                             linestyle='none', marker='o', markersize=5,
                             color='black',mfc='none')
    fig.suptitle('Simple Scatterplot Matrix')
    plt.show()

def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    print(data)
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
        
    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)
    
    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')
    
    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

from pandas.plotting import scatter_matrix
def plot2_scatter_matrix():
    df = pd.read_csv('./data/fuel.csv')
    labels = ['Litres','City','Highway','Carbon']
    data = df[labels]
    sm = scatter_matrix(data, alpha=0.4, figsize=(6, 6), diagonal=" ")
    [s.xaxis.label.set_visible(False) for s in sm.reshape(-1)]
    [s.yaxis.label.set_visible(False) for s in sm.reshape(-1)]
    for i, label in enumerate(labels):
        sm[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                         ha='center', va='center')
    plt.show()

##LagPlot------------------------------------
def lag_plot():
    names = ['qtr','beer']
    df = pd.read_csv('./data/ausbeer.csv',header=None,names=names)
    df['qtr'] = df['qtr'].apply(lambda x:t2dt(x))
    df['qtr'] = df['qtr'].dt.to_period("Q")
    df.set_index(['qtr'], inplace=True)
    df = df.loc['1992Q1':]
    col_names, lag = [], 9
    for i in range(1,lag+1):
        coln = 'lag'+str(i)
        col_names.append(coln)
        df[coln] = df['beer'].shift(i)
        pd.concat([df,df[coln]],axis=1)
    fig, ax = plt.subplots(3, 3, figsize=(8,8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    lst = [(i,j) for i in range(3) for j in range(3)]
    for (indx, name) in zip(lst,col_names):
        df.plot.scatter(x=name,y='beer',ax=ax[indx])
        ax[indx].set_yticklabels([])
        #ax[indx].plot(ax[indx].get_xlim(), ax[indx].get_ylim(), "r--")
        ax[indx].plot([0,1],[0,1],'r--',transform=ax[indx].transAxes)
        ##Using transform=ax.transAxes, the supplied x and y coordinates
        ##are interpreted as axes coordinates instead of data coordinates.
    plt.tight_layout()
    plt.show()

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
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(series,ax=ax[1,0])
    plt.tight_layout()
    plt.show()

##SimpleForecast----------------------------
def simple_forecast():
    names = ['qtr','beer']
    df = pd.read_csv('./data/ausbeer.csv',header=None,names=names)
    df['qtr'] = df['qtr'].apply(lambda x:t2dt(x))
    df['qtr'] = df['qtr'].dt.to_period("Q")
    df.set_index(['qtr'], inplace=True)
    df = df.loc['1992':'2006']
    ax = df.plot()
    ax.set_xlim(df.index[0],df.index[-1]+11)
    plt.show()

##Main--------------------------------------
if __name__ == "__main__":
    #melsyd_plot()
    #a10_plot()
    #season_plot()
    month_plot()
    #test_seasonal_plot()
    #scatter_plot()
    #plot_scatter_matrix()
    #plot2_scatter_matrix()
    #plot_noise()
    #lag_plot()
    #simple_forecast()
