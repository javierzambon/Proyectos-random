import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Pedir al usuario el ticker de la acción y la fecha a predecir
ticker = input("Introduce el ticker de la acción (por ejemplo, NVDA): ").upper()
prediction_date = input("Introduce la fecha que deseas predecir (YYYY-MM-DD): ")

# Validar la fecha introducida
try:
    prediction_date = datetime.strptime(prediction_date, "%Y-%m-%d")
    today = datetime.today()
    if prediction_date <= today:
        raise ValueError("La fecha de predicción debe ser una fecha futura.")
except ValueError as e:
    print(f"Error en la fecha introducida: {e}")
    exit()

# Descargar datos históricos hasta hoy
data = yf.download(ticker, start='2023-01-01', end=today.strftime('%Y-%m-%d'))

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
plt.title(f'Modelo LSTM - Predicción de precios de acciones de {ticker}')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado (USD)')
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Entrenamiento', 'Validación', 'Predicciones'], loc='lower right')
plt.show()

# Realizar la predicción para la fecha especificada
days_to_predict = (prediction_date - today).days
input_data = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))

for _ in range(days_to_predict):
    predicted_price_scaled = model.predict(input_data)
    input_data = np.append(input_data[:, 1:, :], [[predicted_price_scaled]], axis=1)

predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

print(f"El valor predicho para la acción {ticker} el {prediction_date.strftime('%Y-%m-%d')} es: ${predicted_price:.2f}")

####################### Este codigo hace lo siguiente ###########################
# Pide al usuario el ticker de la acción y la fecha que desea predecir.
# Valida la fecha introducida.
# Descarga los datos históricos hasta hoy.
# Entrena el modelo de predicción con esos datos.
# Realiza la predicción para la fecha especificada y muestra el resultado.
