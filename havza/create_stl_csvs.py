import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def fill_and_save(file_in, file_out):
    df = pd.read_csv(file_in)
    df['time'] = pd.to_datetime(df['time'])
    basin_name = df['basin'].iloc[0]
    
    df.set_index('time', inplace=True)
    df = df.resample('MS').mean(numeric_only=True)
    
    is_nan = df['twsa'].isna()
    if is_nan.any():
        df_temp = df['twsa'].interpolate(method='linear')
        stl = STL(df_temp, robust=True, period=12)
        res = stl.fit()
        stl_impute_values = res.trend + res.seasonal
        df['twsa_filled'] = df['twsa'].copy()
        df.loc[is_nan, 'twsa_filled'] = stl_impute_values[is_nan]
    else:
        df['twsa_filled'] = df['twsa']
        
    df.reset_index(inplace=True)
    df['basin'] = basin_name
    
    # Sadece gerekli sütunları kaydet: basin, time, twsa_filled (adını twsa yapalım ki LSTM kodları bozulmasın)
    df_out = df[['basin', 'time', 'twsa_filled']].copy()
    df_out.rename(columns={'twsa_filled': 'twsa'}, inplace=True)
    
    df_out.to_csv(file_out, index=False)
    print(f"Kaydedildi: {file_out}")

fill_and_save('Firat_Dicle_Havzasi_twsa.csv', 'Firat_Dicle_Havzasi_twsa_stl.csv')
fill_and_save('Konya_Kapali_Havzasi_twsa.csv', 'Konya_Kapali_Havzasi_twsa_stl.csv')
print("İşlem tamam!")
