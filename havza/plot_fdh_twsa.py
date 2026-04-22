import pandas as pd
import matplotlib.pyplot as plt

# Veriyi oku
dosya_adi = 'Firat_Dicle_Havzasi_twsa.csv'
print(f"{dosya_adi} dosyası okunuyor...")
df = pd.read_csv(dosya_adi)

# 'time' sütununu datetime formatına çevir ki x ekseninde düzgün görünsün
df['time'] = pd.to_datetime(df['time'])

# -------------------------------------------------------------
# DİKKAT: Uydu verisindeki (özellikle 2017-2018 arası) eksikleri grafikte boşluk olarak 
#         görebilmek için zaman serisindeki atlayan aylara NaN (boş veri) ekleyelim.
df.set_index('time', inplace=True)
df = df.resample('MS').asfreq()  # 'MS' = Her ayın ilk günü. Eksik aylar NaN olacak.
df.reset_index(inplace=True)
# -------------------------------------------------------------

# Grafiği çizdir
plt.figure(figsize=(14, 7))
plt.plot(df['time'], df['twsa'], marker='', linestyle='-', color='b', linewidth=2, label='TWSA (Karasal Su Depolama)')

# Görsel düzenlemeler
plt.title('Fırat-Dicle Havzası TWSA Değişimi (2002 - 2026)', fontsize=16, fontweight='bold')
plt.xlabel('Zaman (Yıl)', fontsize=12)
plt.ylabel('TWSA Değeri (cm)', fontsize=12)
plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='Sıfır Çizgisi (Referans)')
plt.grid(True, alpha=0.4, linestyle=':')
plt.legend(fontsize=12)

# Tarih formatının x ekseninde üst üste binmemesi için
plt.xticks(rotation=45)
plt.tight_layout()

# Grafiği göster ve kaydet
kayit_adi = 'Firat_Dicle_Havzasi_twsa_grafik.png'
plt.savefig(kayit_adi, dpi=300)
print(f"Grafik çizildi ve '{kayit_adi}' adıyla kaydedildi.")

# Ekranda göster
plt.show()
