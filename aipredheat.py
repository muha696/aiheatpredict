import os
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

# --- Выходные/рабочие дни ---
holidays = ['01.01.2023', '07.01.2023', '08.03.2023', '24.04.2023', '25.04.2023', '01.05.2023', '08.05.2023',
            '09.05.2023', '03.07.2023', '06.11.2023', '07.11.2023', '25.12.2023',
            '01.01.2024', '02.01.2024', '08.03.2024', '01.05.2024', '09.05.2024', '13.05.2024', '14.05.2024',
            '03.07.2024', '07.11.2024', '08.11.2024', '25.12.2024',
            '01.01.2025', '02.01.2025', '06.01.2025', '07.01.2025', '08.03.2025', '28.04.2025', '29.04.2025',]

work_days = [
    '29.04.2023', '13.05.2023', '11.11.2023',
    '18.05.2024', '16.11.2024',
    '10.01.2025', '26.04.2025',
]

holiday_dates = pd.to_datetime(holidays, format='%d.%m.%Y')
work_dates = pd.to_datetime(work_days, format='%d.%m.%Y')

# --- Начало и конец отопительных периодов ---
heat_end2223 = pd.to_datetime('26.04.2023', dayfirst=True)

heat_start2324 = pd.to_datetime('06.10.2023', dayfirst=True)
heat_end2324 = pd.to_datetime('29.04.2024', dayfirst=True)


heat_start2425 = pd.to_datetime('01.10.2024', dayfirst=True)
heat_end2425 = pd.to_datetime('15.04.2025', dayfirst=True)

heat_off_months = [
    heat_end2223.month,
    heat_end2324.month,
    heat_end2425.month,
]

heat_on_months = [
    heat_start2324.month,
    heat_start2425.month,
]

# --- Загрузка данных для обучения ---
data = pd.read_csv('db15gvs.csv', delimiter=';', decimal=',')

# --- Преобразование даты и время ---
data['datetime'] = [f'{data["date"][i]} {data["time"][i]}' for i in range(len(data['date']))]
data['datetime'] = data['datetime'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S'))
data['minute'] = data['datetime'].dt.minute
data['month'] = data['datetime'].dt.month

# --- Создание базы для обучения ---

new_data = data[(data['minute'] == 0) & (data['q'] > 0)].copy()

# --- Фильтрация данных и добавление дополнительных признаков ---

new_data['day_of_week'] = new_data['datetime'].dt.dayofweek

new_data['is_weekend'] = (new_data['day_of_week'].isin([5, 6])) | (new_data['datetime'].dt.floor('D').isin(holiday_dates)) & ~ (
    new_data['datetime'].dt.floor('D').isin(work_dates)
)
new_data['is_heatperiod'] = (((new_data['datetime'] >= heat_start2324) & (new_data['datetime'] <= heat_end2324))
                             | ((new_data['datetime'] >= heat_start2425) & (new_data['datetime'] <= heat_end2425))
                             | ((new_data['datetime'] <= heat_end2223)))

new_data['is_heatoffmonth'] = new_data['month'].apply(lambda m: int(m in heat_off_months))
new_data['is_heatonmonth'] = new_data['month'].apply(lambda m: int(m in heat_on_months))
new_data = new_data[new_data['month'] != 4].reset_index(drop=True)

new_data['hour'] = new_data['datetime'].dt.hour
z_scores = np.abs(stats.zscore(new_data['q']))
new_data = new_data[z_scores < 3]

new_data['date_only'] = new_data['datetime'].dt.date
counts_per_day = new_data['date_only'].value_counts()
valid_dates = counts_per_day[counts_per_day == 24].index
new_data = new_data[new_data['date_only'].isin(valid_dates)].reset_index(drop=True)
new_data.drop(columns=['date_only'], inplace=True)

# --- Масштабирование данных для обучения ИИ ---
scaler_tair = StandardScaler()
scaler_y = MinMaxScaler()
scaler_day = StandardScaler()
scaler_hour = StandardScaler()
scaler_month = StandardScaler()

new_data['tair_sc'] = scaler_tair.fit_transform(new_data[['tair']])
new_data['hour_sc'] = scaler_hour.fit_transform(new_data[['hour']])
new_data['day_sc'] = scaler_day.fit_transform(new_data[['day_of_week']])
new_data['month_sc'] = scaler_month.fit_transform(new_data[['month']])
new_data['q_sc'] = scaler_y.fit_transform(new_data[['q']])

# --- Учет цикличности дней недели и времени в сутках ---
new_data['week_sin'] = np.sin(2 * np.pi * new_data['day_of_week'] / 7)
new_data['week_cos'] = np.cos(2 * np.pi * new_data['day_of_week'] / 7)
new_data['hour_sin'] = (np.sin(2 * np.pi * new_data['hour']  / 24))
new_data['hour_cos'] = (np.cos(2 * np.pi * new_data['hour']  / 24))

#new_data = new_data[new_data['is_heatperiod'] == False].reset_index(drop=True) #False - обучение для межотопительного, True - отопительного

# --- Сохранение данных для обучения в excel ---
with pd.ExcelWriter('mlp_datalearn.xlsx') as writer:
    new_data.to_excel(writer, sheet_name='Data', index=False)

# --- Подготовка входных и выходных данных для обучения ИИ ---
features = ['tair_sc', 'day_sc', 'hour_sin', 'hour_cos', 'hour_sc', 'week_sin', 'week_cos', 'is_weekend', 'month_sc', 'is_heatoffmonth', 'is_heatonmonth']
X = new_data[features].astype(np.float32).values
y = new_data['q_sc'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# --- Модель обучения ---
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X.shape[1], kernel_regularizer=l2(0.001)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.1))
model.add(Dense(1))  # последний слой без регуляризации

model.compile(optimizer='adam', loss='mse')

# --- Обучение ---
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(X_train, y_train,
                    epochs=300,
                    batch_size=512,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    verbose=1)

# --- Сохранение модели и формулы масштабов в отдельные файлы для использования в других программах ---
model.save('model_mlp_heatpredict.keras')
joblib.dump({'scaler_tair': scaler_tair, 'scaler_day': scaler_day,
             'scaler_hour': scaler_hour, 'scaler_y': scaler_y,
             'scaler_month': scaler_month,},
            'scalers_mlp_heatpredict.pkl')

# --- Визуализация обучения и работы модели ---
y_pred = model.predict(X_test)
y_pred_real = scaler_y.inverse_transform(y_pred).flatten()
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# --- Ошибка ---
mse = mean_squared_error(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)
print(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}')

metrics_df = pd.DataFrame({
    'MSE': [mse],
    'MAE': [mae],
    'R2': [r2],
})

# --- Графики ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('История обучения')
plt.legend()
plt.grid()

losses = {
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss'],
}

plt.subplot(1, 3, 2)
plt.scatter(y_test_real, y_pred_real, alpha=0.3)
plt.plot([min(y_test_real), max(y_test_real)],
         [min(y_test_real), max(y_test_real)], 'r--')
plt.xlabel('Факт q')
plt.ylabel('Прогноз q')
plt.title('Сравнение факта и прогноза')
plt.grid()


plt.subplot(1, 3, 3)
plt.plot([i for i in range(len(y_pred_real))], y_test_real, linestyle='',
         color = 'black', marker='o', markerfacecolor='r', markersize=3)
plt.plot([i for i in range(len(y_pred_real))], y_pred_real, linestyle='',
         color = 'black', marker='o', markerfacecolor='blue', markersize=3)

plt.show()


result = {
    'date': new_data['datetime'].dt.date,
    'time': new_data['datetime'].dt.time,
    'qreal': new_data['q'],
    'qai': scaler_y.inverse_transform(model.predict(X)).flatten(),

}


with pd.ExcelWriter('result_ailearn.xlsx') as writer:
    pd.DataFrame(result).to_excel(writer, sheet_name='Test', index=False)
    pd.DataFrame(losses).to_excel(writer, sheet_name='Train', index=False)
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)










