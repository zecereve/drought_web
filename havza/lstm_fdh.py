import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Sabitlik ataması
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ----------------- HAVZA SEÇİMİ ----------------- #
# İstediğiniz havzayı burada belirtebilirsiniz. 
BASIN_PREFIX = "Firat_Dicle_Havzasi" 
# BASIN_PREFIX = "Konya_Kapali_Havzasi"

TWSA_PATH = f"{BASIN_PREFIX}_twsa_stl.csv"
ERA5_PATH = f"{BASIN_PREFIX}_era5.csv"

WINDOW_SIZE = 24
EPOCHS = 100
FORECAST_STEPS = 60
START_DATE = "2025-04-01"
MODEL_NAME = "LSTM"

def load_data():
    df_twsa = pd.read_csv(TWSA_PATH)
    df_precip = pd.read_csv(ERA5_PATH)
    
    df_twsa['time'] = pd.to_datetime(df_twsa['time'])
    df_twsa.set_index('time', inplace=True)
    
    df_precip['time'] = pd.to_datetime(df_precip['time'])
    df_precip.set_index('time', inplace=True)
    
    # Zaman eksenine göre inner join yapıyoruz (ikisinin de olduğu tarihler)
    df = df_twsa[['twsa']].join(df_precip[['precip']], how='inner')
    
    data = df[['twsa', 'precip']].copy()
    data['twsa'] = data['twsa'].interpolate(method='time')
    data['precip'] = data['precip'].interpolate(method='time')
    return data

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:(i + window_size), :])
        y.append(dataset[i + window_size, 0]) # Predict TWSA
    return np.array(X), np.array(y)

def build_model(window_size):
    model = Sequential()
    if MODEL_NAME == "LSTM":
        model.add(LSTM(64, return_sequences=False, input_shape=(window_size, 2)))
        model.add(Dropout(0.2))
    elif MODEL_NAME == "GRU":
        model.add(GRU(64, return_sequences=False, input_shape=(window_size, 2)))
        model.add(Dropout(0.2))
    elif MODEL_NAME == "HYBRID":
        model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 2)))
        model.add(Dropout(0.2))
        model.add(GRU(32))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_future(model, last_window, steps, scaler, df_ts, target_date):
    predictions = []
    current_batch = last_window.reshape((1, WINDOW_SIZE, 2))
    
    monthly_precip_avg = df_ts['precip'].groupby(df_ts.index.month).mean()
    future_dates = pd.date_range(start=target_date + pd.DateOffset(months=1), periods=steps, freq='MS')

    for i in range(steps):
        current_pred_twsa = model.predict(current_batch, verbose=0)[0][0]
        predictions.append(current_pred_twsa)
        
        next_month = future_dates[i].month
        next_precip_raw = monthly_precip_avg[next_month]
        
        raw_row = np.array([[0, next_precip_raw]])
        scaled_precip = scaler.transform(raw_row)[0, 1]
        
        new_row_scaled = np.array([[[current_pred_twsa, scaled_precip]]])
        current_batch = np.append(current_batch[:, 1:, :], new_row_scaled, axis=1)

    dummy_pred_array = np.zeros((steps, 2))
    dummy_pred_array[:, 0] = predictions
    inv_predictions = scaler.inverse_transform(dummy_pred_array)[:, 0]
    return inv_predictions

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    nse = 1 - (np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    min_val = min(np.min(y_true), np.min(y_pred))
    amplitude = max(np.max(y_true), np.max(y_pred)) - min_val
    shift_val = abs(min_val) + amplitude if min_val < 0 else amplitude
    
    y_true_shifted = y_true + shift_val
    y_pred_shifted = y_pred + shift_val
    
    smape = 100 / len(y_true_shifted) * np.sum(2 * np.abs(y_pred_shifted - y_true_shifted) / (np.abs(y_true_shifted) + np.abs(y_pred_shifted) + 1e-10))
    return rmse, r2, nse, smape, mae

def main():
    print(f"Pandas: {pd.__version__} | TensorFlow: {tf.__version__}")
    print(f"Havza: {BASIN_PREFIX}")

    if not os.path.exists(TWSA_PATH) or not os.path.exists(ERA5_PATH):
        print(f"Hata: Veri dosyaları bulunamadı! {TWSA_PATH} veya {ERA5_PATH} eksik.")
        return

    df_ts = load_data()

    dataset = df_ts.values
    scaled_data, scaler = prepare_data(dataset)

    X, y = create_sequences(scaled_data, WINDOW_SIZE)

    model = build_model(WINDOW_SIZE)
    model.fit(X, y, epochs=EPOCHS, batch_size=16, verbose=1)

    train_predict = model.predict(X)
    
    dummy_train_pred = np.zeros((len(train_predict), 2))
    dummy_train_pred[:, 0] = train_predict[:, 0]
    train_predict_inv = scaler.inverse_transform(dummy_train_pred)[:, 0]
    
    dummy_y = np.zeros((len(y), 2))
    dummy_y[:, 0] = y
    y_train_inv = scaler.inverse_transform(dummy_y)[:, 0]

    rmse, r2, nse, smape, mae = calculate_metrics(y_train_inv, train_predict_inv)

    print("\nMODEL METRİKLERİ")
    print(f"RMSE:{rmse:.4f} R2:{r2:.4f} NSE:{nse:.4f} SMAPE:{smape:.2f}% MAE:{mae:.4f}")

    print("\nForecast başlatılıyor...")

    forecast_start_date = pd.Timestamp(START_DATE)
    target_date = forecast_start_date - pd.DateOffset(months=1)

    try:
        idx_target = df_ts.index.get_loc(target_date)
    except KeyError:
        print(f"HATA: {target_date.date()} dataset içinde yok!")
        return

    last_window_data = df_ts.iloc[idx_target - WINDOW_SIZE + 1 : idx_target + 1]
    scaled_last_window = scaler.transform(last_window_data[['twsa', 'precip']].values)

    future_predictions = forecast_future(model, scaled_last_window, FORECAST_STEPS, scaler, df_ts, target_date)

    future_dates = pd.date_range(start=forecast_start_date, periods=FORECAST_STEPS, freq='MS')
    future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['twsa_forecast'])

    # Grafikte kıyaslama için tüm gerçek TWSA verisini yeniden okuyalım
    df_full_twsa = pd.read_csv(TWSA_PATH)
    df_full_twsa['time'] = pd.to_datetime(df_full_twsa['time'])
    df_full_twsa.set_index('time', inplace=True)
    df_full_twsa['twsa'] = df_full_twsa['twsa'].interpolate(method='time')

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(df_full_twsa.index, df_full_twsa['twsa'], color="black", linewidth=2.8, alpha=0.55, label=f"Observed TWSA ({BASIN_PREFIX})")
    
    train_dates = df_ts.index[WINDOW_SIZE:]
    ax.plot(train_dates, train_predict_inv, label='Model Fit (Eğitim)', color='gray', alpha=0.4)

    ax.plot(future_df.index, future_df['twsa_forecast'], color="green", linewidth=3, label='Forecast')

    ax.axvline(forecast_start_date, linestyle="--", linewidth=2, color="gray", label=f"Start: {forecast_start_date.strftime('%m.%Y')}")
    ax.set_title(f"TWSA Forecast (Start: {forecast_start_date.strftime('%m.%Y')}) - {MODEL_NAME} - {BASIN_PREFIX.replace('_', ' ')}")
    ax.set_xlabel("Time")
    ax.set_ylabel("TWSA (cm)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    metrics_text = f"TRAIN FIT METRICS\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nsMAPE: {smape:.2f}%"
    overlap_idx = future_df['twsa_forecast'].dropna().index.intersection(df_ts.index)
    if len(overlap_idx) > 0:
        y_true_overlap = df_ts.loc[overlap_idx, 'twsa'].values
        y_pred_overlap = future_df.loc[overlap_idx, 'twsa_forecast'].values
        try:
            o_rmse, o_r2, o_nse, o_smape, o_mae = calculate_metrics(y_true_overlap, y_pred_overlap)
            metrics_text += f"\n\nTEST (OVERLAP) METRICS (n={len(overlap_idx)})\nMAE: {o_mae:.4f}\nRMSE: {o_rmse:.4f}\nR²: {o_r2:.4f}\nsMAPE: {o_smape:.2f}%"
        except:
            pass

    ax.text(
        0.985, 0.985, metrics_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.9, edgecolor="gray")
    )
    
    try:
        ax.set_xlim([pd.to_datetime("2020-01-01"), future_df.index.max()])
    except:
        pass
    fig.tight_layout()

    file_prefix = f"forecast_{MODEL_NAME.lower()}_{BASIN_PREFIX}_{forecast_start_date.strftime('%Y_%m')}"
    future_df.to_csv(f"{file_prefix}.csv")

    fig.savefig(f"{file_prefix}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
