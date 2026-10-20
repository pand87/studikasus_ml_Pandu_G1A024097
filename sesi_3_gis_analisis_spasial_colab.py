# -*- coding: utf-8 -*-
"""
SESI 3: CAREER AND COMPETITION IN INDUSTRIAL ERA WITH GIS
Narasumber : Bapak Widi Primadeva, S.Kom. (GIS Asset Management Analyst, City Care Property NZ)
Studi Kasus: Manajemen Aset Kota Berbasis Geospasial ("Snap Send Solve") & Analisis
             Buffer Radius Penempatan Ritel Baru di Kota Bengkulu.
Platform   : Google Colab (Mandiri / Standalone)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Menghitung jarak permukaan bumi (meter) menggunakan formula Haversine."""
    r_earth = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


print("=== SESI 3: ANALISIS SPASIAL GIS (MANAJEMEN ASET KOTA & BUFFER RITEL) ===")

# 1. Manajemen Aset Kota ("Snap Send Solve") di Bengkulu
df_assets = pd.DataFrame({
    "ticket_id": ["SSS-101", "SSS-102", "SSS-103", "SSS-104", "SSS-105", "SSS-106"],
    "kategori_aset": [
        "Pohon Rawan Tumbang",
        "Drainase Tersumbat",
        "Lampu Taman Mati",
        "Pohon Rawan Tumbang",
        "Alih Fungsi Lahan Sawah",
        "Perawatan Rumput Lansia (Berms)",
    ],
    "lokasi": [
        "Jl. Suprapto",
        "Simpang Lima Ratu Samban",
        "Taman Pantai Panjang",
        "Kampus UNIB Kandang Limun",
        "Lahan Pangan Bengkulu Utara",
        "Permukiman Padang Harapan",
    ],
    "lat": [-3.7928, -3.7956, -3.8062, -3.7592, -3.4310, -3.8015],
    "lon": [102.2608, 102.2655, 102.2630, 102.2735, 102.1850, 102.2810],
    "tingkat_bahaya": [9, 8, 4, 9, 10, 5],
})

depo_lat, depo_lon = -3.7880, 102.2680
df_assets["jarak_dari_depo_km"] = np.round(
    haversine_distance_m(depo_lat, depo_lon, df_assets["lat"], df_assets["lon"]) / 1000.0, 2
)
df_assets["skor_prioritas"] = np.round(
    df_assets["tingkat_bahaya"] / (df_assets["jarak_dari_depo_km"] + 0.5), 2
)
df_assets = df_assets.sort_values(by="skor_prioritas", ascending=False)

print("[1] Urutan Prioritas Penanganan Aset Kota ('Snap Send Solve'):")
print(df_assets[["ticket_id", "kategori_aset", "lokasi", "jarak_dari_depo_km", "skor_prioritas"]].to_string(index=False))

# 2. Analisis Buffer Radius Ritel Baru (Mencegah Kanibalisasi Pasar < 400 Meter)
existing_stores = pd.DataFrame({
    "store_id": ["Gerai-A (UNIB Depan)", "Gerai-B (UNIB Belakang)", "Gerai-C (Pematang Gubernur)"],
    "lat": [-3.7585, -3.7615, -3.7680],
    "lon": [102.2710, 102.2760, 102.2820],
})
kandidat_baru = {"nama": "Kandidat Gerai Baru", "lat": -3.7598, "lon": 102.2725}

existing_stores["jarak_ke_kandidat_m"] = np.round(
    haversine_distance_m(kandidat_baru["lat"], kandidat_baru["lon"], existing_stores["lat"], existing_stores["lon"]), 1
)
min_dist = existing_stores["jarak_ke_kandidat_m"].min()
status_lokasi = (
    "DITOLAK (Terlalu Dekat < 400m / Kanibalisasi Pasar)"
    if min_dist < 400
    else "DISETUJUI (Aman dari Radius Kanibalisasi)"
)

print(f"\n[2] Evaluasi Spasial Kandidat Gerai Ritel Baru (Radius Minimum 400m):")
print(existing_stores.to_string(index=False))
print(f"    -> Jarak Terdekat: {min_dist:.1f} meter | Keputusan: {status_lokasi}")

# Visualisasi Mandiri Peta Koordinat & Buffer Sesi 3
plt.figure(figsize=(7.5, 5))
plt.scatter(existing_stores["lon"], existing_stores["lat"], c="#2563EB", s=120, label="Gerai Eksisting")
plt.scatter(kandidat_baru["lon"], kandidat_baru["lat"], c="#DC2626", marker="X", s=160, label="Kandidat Gerai Baru (Ditolak)")
for _, row in existing_stores.iterrows():
    plt.annotate(row["store_id"], (row["lon"] + 0.0003, row["lat"] + 0.0002), fontsize=8.5)
plt.annotate(kandidat_baru["nama"], (kandidat_baru["lon"] + 0.0003, kandidat_baru["lat"] - 0.0005), fontsize=9, fontweight="bold", color="#DC2626")
plt.title("Sesi 3: Peta Spasial Evaluasi Lokasi Gerai Ritel Sekitar Kampus UNIB", fontweight="bold")
plt.xlabel("Longitude (Bujur Timur)")
plt.ylabel("Latitude (Lintang Selatan)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("output_sesi_3_gis_spasial.png", dpi=150)
plt.show()
plt.close()