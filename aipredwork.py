import os
import joblib

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.models import load_model


# --- Загрузка обученной модели и масшабирования ---
model = load_model('model_mlp_heatpredict.keras')
scaler = joblib.load('scalers_mlp_heatpredict.pkl')
scaler_day = scaler['scaler_day']
scaler_hour = scaler['scaler_hour']
scaler_tair = scaler['scaler_tair']
scaler_y = scaler['scaler_y']
scaler_month = scaler['scaler_month']

# --- Загрузка исходных данных и обработка ---
data = pd.read_csv('test.csv', delimiter=';', decimal=',')
data['datetime'] = [f'{data["date"][i]} {data["time"][i]}' for i in range(len(data['date']))]
data['datetime'] = data['datetime'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'))

data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek

# --- Получение признаков ---
data['is_heatperiod'] = 0
data['is_weekend'] = 0
data['is_heatoffmonth'] = 0
data['is_heatonmonth'] = 0

data['tair_sc'] = scaler_tair.transform(data[['tair']])
data['day_sc'] = scaler_day.transform(data[['day_of_week']])
data['hour_sc'] = scaler_hour.transform(data[['hour']])
data['month_sc'] = scaler_month.transform(data[['month']])

data['week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
data['hour_sin'] = (np.sin(2 * np.pi * data['hour']  / 24))
data['hour_cos'] = (np.cos(2 * np.pi * data['hour']  / 24))

features = ['tair_sc', 'day_sc', 'hour_sin', 'hour_cos', 'hour_sc', 'week_sin', 'week_cos',
            'is_weekend', 'month_sc', 'is_heatoffmonth', 'is_heatonmonth']

X = data[features].astype(np.float32).values
y = model.predict(X)
y_real = scaler_y.inverse_transform(y).flatten()

prediction = {
    'date': data['datetime'].dt.date,
    'time': data['datetime'].dt.time,
    'tair': data['tair'],
    'qai': y_real
}

with pd.ExcelWriter('result_ai_predisction.xlsx') as writer:
    pd.DataFrame(prediction).to_excel(writer, sheet_name='Result', index=False)



