import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Descargar datos históricos de Nvidia
ticker = 'NVDA'
data = yf.download(ticker, start='2023-01-01', end='2024-07-17')

# Usar solo la columna de cierre ajustado
data = data[['Adj Close']]

# Escalar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Crear secuencias de entrenamiento
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Dividir en datos de entrenamiento y prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predecir los precios futuros
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Crear un DataFrame para las predicciones
valid = data[split + sequence_length:]
valid['Predictions'] = predictions

# Mostrar las predicciones
train = data[:split + sequence_length]

plt.figure(figsize=(16, 8))
plt.title('Modelo LSTM - Predicción de precios de acciones de Nvidia')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado (USD)')
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Entrenamiento', 'Validación', 'Predicciones'], loc='lower right')
plt.show()


