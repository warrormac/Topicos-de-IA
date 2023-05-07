import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
import datetime

# Obtener datos de Yahoo Finance
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2020, 12, 31)
stocks = ['AAPL', 'AMZN', 'GOOGL', 'MSFT']
df = yf.download(stocks, start=start_date, end=end_date)

# Obtener datos de índices de Yahoo Finance
index_df = yf.download('^GSPC ^DJI', start=start_date, end=end_date)

# Unir los datos en un solo DataFrame
data = pd.DataFrame()
data['AAPL'] = df['Adj Close']['AAPL']
data['AMZN'] = df['Adj Close']['AMZN']
data['GOOGL'] = df['Adj Close']['GOOGL']
data['MSFT'] = df['Adj Close']['MSFT']
data['S&P500'] = index_df['Adj Close']['^GSPC']
data['DowJones'] = index_df['Adj Close']['^DJI']

# Realizar la transformación de series de tiempo con un time delay de 5 días
lagged_data = pd.DataFrame()
for i in range(5):
    lagged_data[f'AAPL_Lag{i+1}'] = data['AAPL'].shift(i+1)
    lagged_data[f'AMZN_Lag{i+1}'] = data['AMZN'].shift(i+1)
    lagged_data[f'GOOGL_Lag{i+1}'] = data['GOOGL'].shift(i+1)
    lagged_data[f'MSFT_Lag{i+1}'] = data['MSFT'].shift(i+1)
    lagged_data[f'S&P500_Lag{i+1}'] = data['S&P500'].shift(i+1)
    lagged_data[f'DowJones_Lag{i+1}'] = data['DowJones'].shift(i+1)

lagged_data['AAPL'] = data['AAPL']  # Variable dependiente

# Eliminar filas con valores nulos generados por el time delay
lagged_data.dropna(inplace=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X = lagged_data.drop('AAPL', axis=1)  # Variables independientes
y = lagged_data['AAPL']  # Variable dependiente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y ajustar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Prepare new data for prediction
new_data = pd.DataFrame()
new_data['AAPL'] = df['Adj Close']['AAPL'][-5:].values
new_data['AMZN'] = df['Adj Close']['AMZN'][-5:].values
new_data['GOOGL'] = df['Adj Close']['GOOGL'][-5:].values
new_data['MSFT'] = df['Adj Close']['MSFT'][-5:].values
new_data['S&P500'] = index_df['Adj Close']['^GSPC'][-5:].values
new_data['DowJones'] = index_df['Adj Close']['^DJI'][-5:].values

# Add lagged features for the new data
for i in range(5):
    new_data[f'AAPL_Lag{i+1}'] = lagged_data[f'AAPL_Lag{i+1}'].shift(1)[-5:].values
    new_data[f'AMZN_Lag{i+1}'] = lagged_data[f'AMZN_Lag{i+1}'].shift(1)[-5:].values
    new_data[f'GOOGL_Lag{i+1}'] = lagged_data[f'GOOGL_Lag{i+1}'].shift(1)[-5:].values
    new_data[f'MSFT_Lag{i+1}'] = lagged_data[f'MSFT_Lag{i+1}'].shift(1)[-5:].values
    new_data[f'S&P500_Lag{i+1}'] = lagged_data[f'S&P500_Lag{i+1}'].shift(1)[-5:].values
    new_data[f'DowJones_Lag{i+1}'] = lagged_data[f'DowJones_Lag{i+1}'].shift(1)[-5:].values

# Ensure the column order of the new data matches the training data
new_data = new_data[X.columns]

# Make predictions on the new data
predictions = model.predict(new_data)

# Display the predicted prices
print('Predicted Prices:')
for i, pred in enumerate(predictions):
    print(f"Day {i+1}: {pred}")
