from statsmodels.regression.rolling import RollingOLS
import pandas_datareader as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import warnings
import pandas_ta 
from sklearn.cluster import k_means

warnings.filterwarnings('ignore') #ignorar os erros

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

#print(sp500)

sp500['Symbol'] = sp500['Symbol'].str.replace(".","-")

symbols_list = sp500['Symbol'].unique().tolist()

end_date= '2025-01-01'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

print(start_date)

df=yf.download(tickers=symbols_list,start=start_date,end=end_date).stack() #e necessario stackar o multiindex

df.index.names = ['date', 'ticker']

df.columns

print(df)

df['German_Klass_vol'] = (((np.log(df['High']))-np.log(df['Low']))**2)/2-(2*np.log(2)-1)*((np.log(df['Adj Close'])-np.log(df['Open'])))**2

df['rsi'] = df.groupby(level=1)['Adj Close'].transform(lambda x:pandas_ta.rsi(close=x,length=20))

appl_data = df.xs('AAPL',level=1)['rsi'].plot()

appl_data.plot(title="RSI para AAPL", figsize=(10, 6))

#plt.xlabel("Data")
#plt.ylabel("RSI")

#plt.show()

df['bb_low'] = df.groupby(level=1)['Adj Close'].transform(lambda x : pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])

df['bb_mid'] = df.groupby(level=1)['Adj Close'].transform(lambda x : pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])

df['bb_high'] = df.groupby(level=1)['Adj Close'].transform(lambda x : pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['High'],low=stock_data['Low'],close=stock_data['Close'],length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr']=df.groupby(level=1,group_keys=False).apply(compute_atr)

def compute_macd(close):
    macd =pandas_ta.macd(close=close,length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd']=df.groupby(level=1,group_keys=False)['Adj Close'].apply(compute_macd)

df['dollar_volume'] = (df['Adj Close']*df['Volume'])/1e6

last_columns=[c for c in df.columns.unique(0) if c not in ['dollar_volume','Close','Open','High','Low','Volume']]

data = pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('Ticker').to_frame('dollar_volume'),
                df.unstack('ticker')[last_columns].resample('M').mean().stack('Ticker')],axis=1).dropna()

print(data)

data['dollar_volume']=(data.loc[:,'dollar_volume'].unstack('ticker').rolling(5*12).mean().stack()) #trabalhando apenas com dollar volume

data['dollar_volume_rank'] = (data.groupby('Date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_volume_rank']<150].drop(['dollar_volume','dollar_volume_rank'],axis=1) #SO PEGA AS 150 MAIS VOLATEIS

print(data)

g = df.xs('AAPL',level=1)

def calculate_returns(df):

    outlier_cutoff= 0.005

    lags = [1,2,3,6,9,12]

    #x.clip limitar entre 2 valores 
    #pow é igual a raiz do resultado
    for lag in lags:

       df[f'return_{lag}m'] = (df['Adj Close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
       
    return df
    #funcionamento da funçao é que ela pega o adj close da AAPL pega a mudanca percentual e depois faz o rendimento anulizado
    #pow = elevar a ,no caso elevou 1/lag que é raiz de lag

data = data.groupby(level=1,group_keys=False).apply(calculate_returns).dropna()

print(data)

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data=factor_data.resample('M').last().div(100) 

data.index = data.index.set_levels(
    [level.tz_localize(None) if hasattr(level, 'tz_localize') else level for level in data.index.levels])

factor_data=factor_data.join(data['return_1m']).sort_index()

print(factor_data)

observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10] #filtro de stocks com mais de 10 meses de dados

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

betas =(factor_data.groupby(level = 1,group_keys=False)
.apply(lambda x : RollingOLS(endog=x['return_1m'],exog=sm.add_constant(x.drop('return_1m',axis=1)),window=min(24,x.shape[0]),min_nobs=(len(x.columns)+1))
.fit(params_only=True)
.params
.drop('const',axis=1)))

print(betas)

data = data.join(betas.groupby('ticker').shift())

factors = ['Mkt-RF','SMB','HML','RMW','CMA']

data.loc[:,factors] = data.groupby('ticker',group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

print(data)

print(data)

data = data.drop('cluster',axis=1)

def Get_Cluster(df):
    df['cluster'] = k_means(n_clusters=4,random_state=0,init='random').fit(df).labels__
    return df

data = data.dropna().groupby('date',group_keys=False)

print(data)