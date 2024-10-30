import numpy as np


# Kullanıcının girdiği sayıları sınıflara ayır
def get_labels_from_console():
    # Kullanıcıdan gerçek etiketleri (Ytest) ve tahmin edilen etiketleri (Ytahmin) al
    sayilar = input("Sınıflar için sırasıyla 6 sayı girin (toplamda 6 sayı, boşlukla ayırın): ").split()

    # Kontrol: Kullanıcıdan 6 sayı girilmiş mi
    if len(sayilar) != 6:
        print("Lütfen toplamda 6 sayı girin.")
        return None, None

    # Sayıları tam sayıya çevir
    sayilar = list(map(int, sayilar))

    # Sınıfları ayır
    sinif1_ytest = sayilar[0]  # Sınıf 1 için gerçek etiket
    sinif1_ytahmin = sayilar[3]  # Sınıf 1 için tahmin edilen etiket
    sinif2_ytest = sayilar[1]  # Sınıf 2 için gerçek etiket
    sinif2_ytahmin = sayilar[4]  # Sınıf 2 için tahmin edilen etiket
    sinif3_ytest = sayilar[2]  # Sınıf 3 için gerçek etiket
    sinif3_ytahmin = sayilar[5]  # Sınıf 3 için tahmin edilen etiket

    return (sinif1_ytest, sinif2_ytest, sinif3_ytest), (sinif1_ytahmin, sinif2_ytahmin, sinif3_ytahmin)


# Metrik hesaplama fonksiyonu
def calculate_metrics(confusion_matrix):
    metrics = {}
    total_samples = confusion_matrix.sum()  # Toplam örnek sayısını hesapla
    num_classes = confusion_matrix.shape[0]

    for i in range(num_classes):  # Her sınıf için döngü
        TP = confusion_matrix[i, i]  # Doğru pozitif tahminler
        FP = confusion_matrix[:, i].sum() - TP  # Yanlış pozitif tahminler
        FN = confusion_matrix[i, :].sum() - TP  # Yanlış negatif tahminler
        TN = total_samples - (TP + FP + FN)  # Doğru negatif tahminler

        # Metrikleri hesapla
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Her sınıf için metrikleri kaydet
        metrics[f"Sınıf {i + 1}"] = {
            "Doğru Pozitif (TP)": TP,
            "Yanlış Pozitif (FP)": FP,
            "Yanlış Negatif (FN)": FN,
            "Doğru Negatif (TN)": TN,
            "Hassasiyet (Precision)": precision,
            "Duyarlılık (Recall)": recall,
            "Özgünlük (Specificity)": specificity
        }

    return metrics


# Ana fonksiyon
def main():
    # Gerçek ve tahmin edilen etiketleri al
    Ytest, Ytahmin = get_labels_from_console()

    # Eğer etiketler alınamadıysa, çık
    if Ytest is None or Ytahmin is None:
        return

    # 3x3 karışıklık matrisi oluştur
    confusion_matrix_3x3 = np.zeros((3, 3), dtype=int)

    # Her sınıf için TP, FP, FN değerlerini hesapla
    for i in range(3):
        true_label = Ytest[i]
        predicted_label = Ytahmin[i]
        confusion_matrix_3x3[true_label, predicted_label] += 1

    print("\nKarışıklık Matrisi (3x3):\n", confusion_matrix_3x3)

    # Metrikleri hesapla ve göster
    metrics = calculate_metrics(confusion_matrix_3x3)
    for sınıf, değerler in metrics.items():
        print(f"\n{sınıf} Metrikleri:")
        for metrik, değer in değerler.items():
            print(f"  {metrik}: {değer:.2f}")


# Programı çalıştır
main()
