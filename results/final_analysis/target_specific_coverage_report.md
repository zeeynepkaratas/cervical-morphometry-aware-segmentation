# Target-Specific Coverage Fragility Report

## 1. Veri doğrulaması

- Kullanılan dosyalar: conformal_results.csv, conformal_per_cell.csv, coverage_gap_analysis.csv, interval_width_failure_analysis.csv, severity_trends.csv, cluster_bootstrap_intervals.csv, degradation_metrics_by_seed_severity.csv, dice_morphometry_spearman.csv, final_analysis_summary.json
- Eksik dosyalar: yok
- Frozen calibration doğrulaması: geçti
- Degradation altında yeniden calibration yapılmadı: geçti
- Cell-wise bağımlılık kontrolü: bootstrap örnekleme birimi `cell_id`, küme sayısı 184.

## 2. N/C sonucu

- Clean coverage: 0.874094
- Degraded coverage: 0.866546
- Ortalama değişim: -0.007548
- >=5 puan düşen koşul: 2
- Seed tutarlılığı: 20260803: PARTIALLY_MORE_FRAGILE, 20260804: PARTIALLY_MORE_FRAGILE, 20260805: PARTIALLY_MORE_FRAGILE
- Severity trendi: cogunlukla_monoton, tam_monoton, cogunlukla_monoton, cogunlukla_monoton
- Interval width / failure etkisi: width mean 1.347880, failure rate 0.000000.

## 3. Circularity sonucu

- Clean coverage: 0.863131
- Degraded coverage: 0.798710
- Ortalama değişim: -0.064420
- >=5 puan düşen koşul: 11
- Seed tutarlılığı: 20260803: PARTIALLY_MORE_FRAGILE, 20260804: PARTIALLY_MORE_FRAGILE, 20260805: PARTIALLY_MORE_FRAGILE
- Severity trendi: cogunlukla_monoton, tam_monoton, cogunlukla_monoton, tam_monoton
- Interval width / failure etkisi: width mean 0.498773, failure rate 0.008907.

## 4. Doğrudan hedef karşılaştırması

- Circularity daha fazla düşen koşullar: 21
- N/C daha fazla düşen koşullar: 14
- Ortalama fark: -0.056872
- Median fark: -0.019111
- Bootstrap %95 güven aralığı: [-0.076796, -0.036110]
- Blur/noise ayrımı: blur 0.015008, noise -0.128752
- Seed tutarlılığı: 20260803: PARTIALLY_MORE_FRAGILE, 20260804: PARTIALLY_MORE_FRAGILE, 20260805: PARTIALLY_MORE_FRAGILE

## 5. Dice sabitken güvenilirlik

- `|delta Dice| <= 0.01` koşul sayısı: 24
- Bu koşullarda N/C coverage değişimi: 0.000226
- Circularity coverage değişimi: -0.001951
- Hedef ayrışması: -0.002178

## 6. Nihai karar

SINIRLI AMA SAVUNULABİLİR SİNYAL

- Circularity gerçekten N/C'den daha kırılgan mı? Kısmen evet
- Fark pratik olarak büyük mü? Evet, belirgin
- Seed'ler arasında tutarlı mı? Kısmen
- Severity ile düzenli mi? Kısmen
- Dice bu farkı açıklıyor mu? Hayır; küçük Dice değişimli koşullarda da hedef farkı ayrıca ölçülebiliyor.
- Conformal kısmı projenin merkezine alınmalı mı? Yardımcı kanıt olarak tutulmalı, tek merkez yapılmamalı.
- Bu sonuç projeyi güçlü hale getiriyor mu? Tek başına güçlü yapmıyor.
- Mevcut proje orta düzeyde mi kalıyor? Evet, mevcut kanıt orta ama savunulabilir düzeyde.
- Yeni projeye pivot gerekli mi? Hayır; seçici daraltma ile devam edilebilir.

## 7. Hocaya sunulacak özet

Frozen-clean conformal çıktılar üzerinden yapılan hedefe özgü kontrolde circularity coverage kaybı N/C'ye göre bazı koşullarda daha yüksek göründü. Ancak farkın büyüklüğü sınırlı ve tüm seed/degradation kombinasyonlarında güçlü biçimde tekrarlanmıyor. Bootstrap belirsizlik aralığı bu ayrımı mutlak bir hedefe özgü kırılganlık iddiasına dönüştürmek için yeterince keskin değilse sonuç temkinli yorumlanmalıdır. Dice değişimi küçük kalan koşullarda da hedef coverage farkı izlenebildiği için yalnız segmentasyon Dice skoru tüm güvenilirlik davranışını açıklamıyor. Bu nedenle conformal analiz proje içinde destekleyici ve seçici bir argüman olarak sunulmalı, ana iddia ise N/C-aware segmentasyonun sınırlı fakat ölçülebilir etkisi üzerine kurulmalıdır.
