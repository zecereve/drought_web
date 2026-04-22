import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def main():
    dosya_adi = 'Kızılırmak_Havzası_twsa.csv'
    csv_cikis_adi = 'Kizilirmak_Havzasi_twsa_stl.csv'
    grafik_cikis_adi = 'Kizilirmak_Havzasi_twsa_stl_grafik.png'
    
    print(f"{dosya_adi} dosyası okunuyor ve eksik veriler STL ile dolduruluyor...")

    # Veriyi oku
    df = pd.read_csv(dosya_adi)
    df['time'] = pd.to_datetime(df['time'])

    # Kızılırmak verisinde mükerrer aylar olduğundan önce ortalama alıyoruz
    df = df.groupby('time').mean()
    
    # Eksik ayları (NaN) barındıran tam indeks yapısına dönüştürüyoruz
    all_months = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
    df = df.reindex(all_months)

    is_nan = df['twsa'].isna()

    if is_nan.any():
        df_temp = df['twsa'].interpolate(method='linear')
        
        # STL Ayrıştırması
        stl = STL(df_temp, robust=True, period=12)
        res = stl.fit()
        
        # Trend + Mevsimsellik tahmini
        stl_impute_values = res.trend + res.seasonal
        
        df['twsa_filled'] = df['twsa'].copy()
        df.loc[is_nan, 'twsa_filled'] = stl_impute_values[is_nan]
    else:
        df['twsa_filled'] = df['twsa']

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)

    # ----------- CSV KAYDETME -----------
    # Sadece istenen formattaki veriyi alalım
    df_out = df[['time', 'twsa_filled']].copy()
    df_out.rename(columns={'twsa_filled': 'twsa'}, inplace=True) 
    df_out.to_csv(csv_cikis_adi, index=False)
    print(f"Veri başarıyla STL ile dolduruldu ve '{csv_cikis_adi}' olarak kaydedildi.")

    # ----------- GRAFİK ÇİZİMİ -----------
    plt.figure(figsize=(14, 7))

    # STL ile doldurulmuş arka plan çizgisi (kesintisiz)
    plt.plot(df['time'], df['twsa_filled'], linestyle='--', color='orange', linewidth=2, alpha=0.8, label='STL ile Doldurulan Boşluklar')

    # Orijinal verinin ana çizgisi (kesikli / gaps)
    plt.plot(df['time'], df['twsa'], marker='o', markersize=4, linestyle='-', color='g', linewidth=2, label='Orijinal TWSA Gözlemi')

    # Özel olarak eksik olup STL ile doldurulan kısımları vurgula
    if is_nan.any():
        plt.scatter(df.loc[is_nan.values, 'time'], df.loc[is_nan.values, 'twsa_filled'], 
                    color='red', s=40, zorder=5, label='Veri Olmayan Aylara STL Tahmini')

    # Görsel ayarlar
    plt.title('Kızılırmak Havzası TWSA Değişimi (STL Doldurma Modeli)', fontsize=16, fontweight='bold')
    plt.xlabel('Zaman (Yıl)', fontsize=12)
    plt.ylabel('TWSA Değeri (cm)', fontsize=12)
    plt.axhline(0, color='red', linestyle='-', alpha=0.4, label='Sıfır Çizgisi')

    plt.grid(True, alpha=0.4, linestyle=':')
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Kaydet
    plt.savefig(grafik_cikis_adi, dpi=300)
    plt.close()
    
    print(f"Grafik başarıyla çizildi ve '{grafik_cikis_adi}' olarak kaydedildi.")

if __name__ == "__main__":
    main()
