import pandas as pd
import ta.momentum
import ta.wrapper
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style="white")

data_inicial='2017-01-01'
data_final='2024-12-15'

raiz4=yf.download('UGPA3.SA',data_inicial,data_final,interval='1d')[['Open','High','Low','Close','Adj Close','Volume']]

dados = raiz4.copy()

for index,row in dados.iterrows():
    dados.at[index,'High']= row['High']/row['Close']*row['Adj Close']
    dados.at[index,'Low']= row['Low']/row['Close']*row['Adj Close']
    dados.at[index,'Open']= row['Open']/row['Close']*row['Adj Close']
    dados.at[index,'Close']= row['Adj Close']


print(dados)

#calculo das medias moveis 

periodo_1=7
periodo_2=14

dados['Curta']=dados['Close'].rolling(periodo_1).mean().bfill()
dados['Longa']=dados['Close'].rolling(periodo_2).mean().bfill()  

#dados[['Close','Curta','Longa']].iloc[0:600].plot(figsize = (15,5))
#plt.show()

#trend following

Operar_vendido=True

dados['Saldo trade']=0
comprado=0
saldo=100

for index,row in dados.iterrows():
    if (row['Curta']>row['Longa']) and (comprado==0):
        comprado=1
        val_compra= row['Close']
    elif (row['Longa']>row['Curta']) and (comprado==1):
        comprado=0
        saldo=saldo*row['Close']/val_compra
    elif (row['Longa']>row['Curta']) and (comprado==0) and (Operar_vendido):
        comprado=-1
        val_venda=row['Close']
    elif (row['Curta']>row['Longa']) and (comprado==-1) and (Operar_vendido):
        comprado=0
        saldo=saldo*val_venda/row['Close']
    dados.at[index,'Saldo trade'] = saldo

dados_plot=dados[['Close','Saldo trade']]/dados[['Close','Saldo trade']].iloc[0]
dados_plot.plot(figsize=(15,5))
#plt.show()

#uso de bolinger bands
import ta.wrapper as taw

ind_BB=taw.BollingerBands(close=dados['Close'],window=20,window_dev=2)

dados['bb_avg']=ind_BB.bollinger_mavg()
dados['bb_high'] = ind_BB.bollinger_hband()
dados['bb_low'] = ind_BB.bollinger_lband()

dados[['Close','bb_high','bb_low','bb_avg']].iloc[0:600].plot(figsize=(15,5))
#plt.show()


dados['Saldo_trade'] = 0
saldo=100
comprado=0
for index,row in dados.iterrows():
    if (row['bb_high']>row['Close']) and (comprado==0):
        comprado=1
        val_compra=row['Close']
    elif(row['Close']>row['bb_avg']) and (comprado==1):
        comprado=0
        saldo=saldo*row['Close']/val_compra
    elif(row['Close']>row['bb_low']) and (comprado==0) and (Operar_vendido == True):
        comprado=-1
        val_venda=row['Close']
    elif(row['Close']<row['bb_avg']) and (comprado==-1) and (Operar_vendido == True):
        comprado=0
        saldo=saldo*val_venda/row['Close']
    dados.at[index,'Saldo_trade'] = saldo

dados_plot_2=dados[['Close','Saldo_trade']]/dados[['Close','Saldo_trade']].iloc[0]
dados_plot_2.plot(figsize=(15,5))
plt.show()

