# -*- coding: utf-8 -*-
"""
SESI 2: BRIDGING GLOBAL INFRASTRUCTURE TO ADDRESS LOCAL CHALLENGE WITH HPC
Narasumber : Asst. Prof. Dr. Worawan Diaz Carballo (Thammasat University, Thailand)
Studi Kasus: Benchmarking Kecepatan Komputasi (GFLOPS) & Ko-Simulasi Berbasis Agen
             (Agent-Based Simulation) Efisiensi Energi Listrik Gedung Kampus.
Platform   : Google Colab (Mandiri / Standalone)
"""

import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("=== SESI 2: HIGH PERFORMANCE COMPUTING (HPC) & SIMULASI ENERGI KAMPUS ===")

# 1. Benchmarking FLOPS (Floating-Point Operations Per Second) ala LINPACK
n_dim = 1000
mat_a = np.random.rand(n_dim, n_dim).astype(np.float64)
mat_b = np.random.rand(n_dim, n_dim).astype(np.float64)

t0 = time.perf_counter()
_ = np.dot(mat_a, mat_b)
elapsed = time.perf_counter() - t0
total_flops = 2 * (n_dim ** 3)
gflops = (total_flops / elapsed) / 1e9
print(f"[1] Benchmark LINPACK MatMul ({n_dim}x{n_dim} FP64): {elapsed:.4f} detik | Performa: {gflops:.2f} GFLOPS\n")

# 2. Agent-Based Simulation Penggunaan Energi Listrik Gedung Kampus (30 Hari)
hours = np.arange(24)
days = 30
outdoor_temp_profile = 26.0 + 6.5 * np.sin((hours - 8) * np.pi / 12)


def simulate_campus_energy(apply_green_policy=False):
    total_kwh = 0.0
    hourly_loads = np.zeros(24)

    for d in range(days):
        is_weekend = (d % 7) in [5, 6]
        for h in hours:
            t_out = outdoor_temp_profile[h] + np.random.normal(0, 0.5)
            if is_weekend:
                active_students = np.random.randint(15, 60)
            elif 8 <= h <= 17:
                active_students = np.random.randint(250, 500)
            elif 18 <= h <= 21:
                active_students = np.random.randint(60, 140)
            else:
                active_students = np.random.randint(0, 15)

            # Lab Biologi wajib menyala 24 jam penuh agar sampel riset tidak rusak
            bio_lab_kw = 18.0

            if not apply_green_policy:
                ac_setpoint = 20.0
                cooling_factor = max(0.0, t_out - ac_setpoint) * 2.8
                general_rooms_kw = (45.0 + cooling_factor + 0.08 * active_students) if (7 <= h <= 22) else 15.0
            else:
                # Kebijakan Smart Campus: AC minimal 25°C (09.00-20.00), kelas/perpustakaan tutup akhir pekan
                ac_setpoint = 25.0
                cooling_factor = max(0.0, t_out - ac_setpoint) * 2.8
                if is_weekend:
                    general_rooms_kw = 4.0
                elif 9 <= h < 20:
                    general_rooms_kw = 28.0 + cooling_factor + 0.06 * active_students
                else:
                    general_rooms_kw = 5.0

            step_kw = bio_lab_kw + general_rooms_kw
            total_kwh += step_kw
            hourly_loads[h] += step_kw / days

    return total_kwh, hourly_loads


kwh_baseline, load_base = simulate_campus_energy(apply_green_policy=False)
kwh_policy, load_policy = simulate_campus_energy(apply_green_policy=True)
saving_pct = ((kwh_baseline - kwh_policy) / kwh_baseline) * 100
tarif_per_kwh = 1444.70

print("[2] Hasil Ko-Simulasi Energi Gedung Kampus (30 Hari):")
print(f"    - Tanpa Kebijakan (Baseline) : {kwh_baseline:,.1f} kWh (Rp {kwh_baseline * tarif_per_kwh:,.0f})")
print(f"    - Kebijakan Smart Campus     : {kwh_policy:,.1f} kWh (Rp {kwh_policy * tarif_per_kwh:,.0f})")
print(f"    - Penghematan Energi & Biaya : {saving_pct:.1f}% (Hemat Rp {(kwh_baseline - kwh_policy) * tarif_per_kwh:,.0f}/bulan)")

# Visualisasi Mandiri Sesi 2
plt.figure(figsize=(8.5, 4.5))
plt.plot(hours, load_base, color="#94A3B8", linestyle="--", linewidth=2, label="Tanpa Kebijakan (AC 20°C)")
plt.plot(hours, load_policy, color="#0D9488", linewidth=2.5, label="Kebijakan Smart Campus (AC ≥ 25°C)")
plt.fill_between(hours, load_policy, load_base, color="#99F6E4", alpha=0.35, label="Energi Dihemat")
plt.title("Sesi 2: Profil Beban Listrik Harian Gedung Kampus (Agent-Based Co-Simulation)", fontweight="bold")
plt.xlabel("Jam Operasional (00:00 - 23:00)")
plt.ylabel("Rata-Rata Beban Daya (kW)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("output_sesi_2_hpc_energi.png", dpi=150)
plt.show()
plt.close()