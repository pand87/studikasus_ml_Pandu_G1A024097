# -*- coding: utf-8 -*-
"""
SESI 5: HUMAN-COMPUTER INTERACTION (HCI) AND DATA VISUALIZATION
Narasumber : Dr. Arafat Febriandirza (Ketua Kelompok Riset HCI & Visualisasi BRIN)
Studi Kasus: Perancangan Dashboard KPI untuk Persona "Budi" (Data Analyst) dengan
             Pre-Attentive Attributes, Prinsip Gestalt, Data-Ink Ratio (Edward Tufte),
             dan Mantra Interaktif Ben Shneiderman.
Platform   : Google Colab (Mandiri / Standalone)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("=== SESI 5: HCI & VISUALISASI DATA (DASHBOARD PERSONA 'BUDI') ===")

# 1. Simulasi Data Pemantauan KPI Harian untuk Persona 'Budi' (30 Hari)
days = np.arange(1, 31)
waktu_muat_detik = np.round(np.random.normal(loc=2.8, scale=0.45, size=30), 2)
# Sisipkan 2 hari anomali kritis (> 8 detik) agar ditangkap Pre-Attentive Attribute (< 200 ms)
waktu_muat_detik[11] = 11.4
waktu_muat_detik[24] = 14.2

wilayah = ["Jakarta Pusat", "Jakarta Selatan", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara"]
kpi_capaian_pct = [94.5, 91.2, 88.0, 62.5, 89.4]  # Jakarta Timur mengalami anomali penurunan KPI

df_kpi_harian = pd.DataFrame({
    "hari_ke": days,
    "waktu_muat_detik": waktu_muat_detik,
    "status_sla": np.where(waktu_muat_detik > 5.0, "ANOMALI KRITIS (>5 dtk)", "Normal (<=5 dtk)"),
})

# 2. Penerapan Mantra Shneiderman: "Overview first, zoom and filter, then detail on demand"
print(f"[1] OVERVIEW : Rata-rata waktu muat = {waktu_muat_detik.mean():.2f} dtk | Total hari anomali = {np.sum(waktu_muat_detik > 5.0)} hari")

df_zoom_filter = df_kpi_harian[df_kpi_harian["waktu_muat_detik"] > 5.0]
print(f"[2] ZOOM & FILTER (Waktu Muat > 5 Detik):")
print(df_zoom_filter.to_string(index=False))

print("\n[3] DETAIL ON DEMAND (Rincian Akar Masalah Anomali):")
for _, row in df_zoom_filter.iterrows():
    print(f"    -> Hari ke-{int(row['hari_ke']):02d} ({row['waktu_muat_detik']} dtk): Lonjakan antrean query tanpa indeks pada server wilayah Jakarta Timur.")

# 3. Visualisasi Dashboard dengan Prinsip Edward Tufte (Data-Ink Ratio Tinggi) & Pre-Attentive Color
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle("Sesi 5: Dashboard KPI Persona 'Budi' (Pre-Attentive Red Alert & High Data-Ink Ratio)", fontweight="bold", fontsize=12)

# Grafik Kiri: Tren Waktu Muat (Warna merah hanya menyala pada titik anomali)
colors_time = ["#DC2626" if v > 5.0 else "#CBD5E1" for v in waktu_muat_detik]
ax1.bar(days, waktu_muat_detik, color=colors_time, width=0.75)
ax1.axhline(5.0, color="#334155", linestyle="--", linewidth=1, label="Batas Maksimal SLA Budi (5 dtk)")
ax1.set_title("Waktu Muat Dashboard Harian (Merah = Anomali)", fontsize=10, fontweight="bold")
ax1.set_xlabel("Hari ke-")
ax1.set_ylabel("Detik")
ax1.legend(frameon=False, loc="upper left")

# Grafik Kanan: Perbandingan Kategori Wilayah (Bar Chart bersih tanpa grid/3D)
colors_reg = ["#DC2626" if v < 75.0 else "#2563EB" for v in kpi_capaian_pct]
bars = ax2.barh(wilayah, kpi_capaian_pct, color=colors_reg, height=0.55)
ax2.set_title("Capaian KPI Harian per Wilayah Jakarta (%)", fontsize=10, fontweight="bold")
ax2.set_xlabel("Persentase Capaian KPI (%)")
ax2.set_xlim(0, 110)
for b in bars:
    w = b.get_width()
    ax2.text(w + 1.5, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("output_sesi_5_hci_dashboard.png", dpi=150)
plt.show()
plt.close()