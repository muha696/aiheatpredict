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


# --- Загрузка обученной модели ---
model = load_model('model_mlp_heatpredict.keras')
scaler = joblib.load('scalers_mlp_heatpredict.pkl')
scaler_day = scaler['scaler_day']
scaler_hour = scaler['scaler_hour']
scaler_tair = scaler['scaler_tair']
scaler_y = scaler['scaler_y']
scaler_month = scaler['scaler_month']

