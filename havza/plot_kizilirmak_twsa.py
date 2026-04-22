import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Dosya yolu
    data_path = "Kızılırmak_Havzası_twsa.csv"
    
    # Veriyi okuma
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby('time').mean()
    
    # Grafikteki boşlukların (missing data) tam olarak kopuk görünmesi için
    # başlangıç ve bitiş aylarını kapsayan kesintisiz bir aylık frekans indeksi oluşturuyoruz.
    all_months = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
    df = df.reindex(all_months)
    
    # Çizim
    plt.figure(figsize=(18, 6))
    
    # Matplotlib NaN değerleri otomatik olarak boşluk (kopuk çizgi) olarak gösterir
    plt.plot(df.index, df['twsa'], marker='o', markersize=4, linestyle='-', color='royalblue', linewidth=1.5, label='Kızılırmak TWSA')
    
    plt.title('Kızılırmak Havzası TWSA Zaman Serisi (Orijinal / Boşluklu Veri)')
    plt.xlabel('Zaman (Yıl-Ay)')
    plt.ylabel('TWSA (cm)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # PNG olarak kaydetme
    output_filename = "Kizilirmak_Havzasi_twsa_bosluklu.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"Grafik başarıyla '{output_filename}' adıyla kaydedildi.")

if __name__ == "__main__":
    main()
