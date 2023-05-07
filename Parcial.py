import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

# Configurar yfinance
yf.pdr_override()

# Definir símbolos de acciones y fechas
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', '^GSPC', '^DJI']
start_date = '2015-01-01'
end_date = '2023-03-31'

# Obtener datos de Yahoo Finance
data = pdr.get_data_yahoo(symbols, start=start_date, end=end_date)

# Seleccionar solo los precios de cierre ajustados
adj_close = data['Adj Close'].copy()

# Agregar los precios de las acciones de Microsoft, Google y Amazon como variables correlacionadas
adj_close.loc[:, 'MSFT'] = data['Adj Close']['MSFT']
adj_close.loc[:, 'GOOGL'] = data['Adj Close']['GOOGL']
adj_close.loc[:, 'AMZN'] = data['Adj Close']['AMZN']

# Agregar los valores de los índices S&P 500 y Dow Jones como variables correlacionadas
sp500_close = data['Adj Close']['^GSPC']
dow_close = data['Adj Close']['^DJI']
adj_close.loc[:, 'S&P500'] = sp500_close
adj_close.loc[:, 'DowJones'] = dow_close

# Verificar si hay datos disponibles para el mes de abril de 2023
if '2023-04-01' in adj_close.index:
    # Obtener los datos del mes de abril de 2023 como conjunto de prueba
    test_data = adj_close.loc['2023-04-01':'2023-04-30']
    # Mostrar el conjunto de datos de prueba
    print(test_data)
else:
    print("No hay datos disponibles para el mes de abril de 2023.")

# Mostrar el conjunto de datos de entrenamiento
print(adj_close)
