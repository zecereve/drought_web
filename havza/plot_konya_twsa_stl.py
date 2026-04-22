import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

dosya_adi = 'Konya_Kapali_Havzasi_twsa.csv'
print(f"{dosya_adi} dosyası okunuyor ve eksik veriler STL ile dolduruluyor...")

# Veriyi oku ve zaman hedefine çevir
df = pd.read_csv(dosya_adi)
df['time'] = pd.to_datetime(df['time'])

# Asıl veriyi eksik aylarla (NaN) genişlet
df.set_index('time', inplace=True)
df = df.resample('MS').mean(numeric_only=True) # Aylık periyot (boşluklar NaN olacak)

# STL işlemi NaN verileri desteklemediği için önce geçici lineer interpolasyon yapıyoruz
# Sonrasında sadece o eksik olan kısımları STL (Trend + Mevsimsellik) tahminiyle değiştireceğiz
is_nan = df['twsa'].isna()

if is_nan.any():
    # Geçici doldurma (Linear), STL'in çalışabilmesi için
    df_temp = df['twsa'].interpolate(method='linear')
    
    # STL Ayrıştırması (12 Aylık Periyot, verideki uç değerlere karşı Robust)
    stl = STL(df_temp, robust=True, period=12)
    res = stl.fit()
    
    # Yeni bir sütunda STL formülü: Trend + Seasonality
    # Kalıntıyı (Residual) bilinmeyen noktalarda 0 sayarak en ideal beklenen değeri elde ederiz
    stl_impute_values = res.trend + res.seasonal
    
    # Orijinal tabloya doldurulmuş seriyi ekle
    df['twsa_filled'] = df['twsa'].copy()
    df.loc[is_nan, 'twsa_filled'] = stl_impute_values[is_nan]
else:
    df['twsa_filled'] = df['twsa']

df.reset_index(inplace=True)

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
plt.title('Konya Kapalı Havzası TWSA Değişimi (STL Doldurma Modeli)', fontsize=16, fontweight='bold')
plt.xlabel('Zaman (Yıl)', fontsize=12)
plt.ylabel('TWSA Değeri (cm)', fontsize=12)
plt.axhline(0, color='red', linestyle='-', alpha=0.4, label='Sıfır Çizgisi')

plt.grid(True, alpha=0.4, linestyle=':')
plt.legend(fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Kaydet
kayit_adi = 'Konya_Kapali_Havzasi_twsa_stl_grafik.png'
plt.savefig(kayit_adi, dpi=300)
print(f"Grafik başarıyla çizildi ve '{kayit_adi}' olarak kaydedildi.")

# Ekranda Göster
plt.show()
