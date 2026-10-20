# -*- coding: utf-8 -*-
"""
SESI 4: INTRODUCTION TO MACHINE LEARNING
Narasumber : Dr. Taniza binti Tajuddin (Universiti Teknologi MARA / UiTM Kedah, Malaysia)
Studi Kasus: Pergeseran Paradigma Rule-Based Thinking ke Data-Driven Thinking
             (Supervised Classification & Unsupervised K-Means Clustering).
Platform   : Google Colab (Mandiri / Standalone)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

np.random.seed(42)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("=== SESI 4: RULE-BASED VS DATA-DRIVEN THINKING & WORKFLOW MACHINE LEARNING ===")

# 1. Simulasi Data Akademik 500 Mahasiswa
n_students = 500
jam_belajar = np.random.uniform(2, 25, size=n_students)
kehadiran_pct = np.random.uniform(55, 100, size=n_students)
skor_tugas = np.clip(40 + 1.8 * jam_belajar + 0.25 * kehadiran_pct + np.random.normal(0, 8, size=n_students), 0, 100)

logit_pass = -6.5 + 0.22 * jam_belajar + 0.05 * kehadiran_pct + 0.04 * skor_tugas - 0.008 * ((jam_belajar - 15) ** 2)
y_true = (1.0 / (1.0 + np.exp(-logit_pass)) > 0.52).astype(int)

df_ml = pd.DataFrame({
    "jam_belajar": jam_belajar,
    "kehadiran_pct": kehadiran_pct,
    "skor_tugas": skor_tugas,
    "lulus_tepat_waktu": y_true,
})

# 2. Rule-Based Thinking (Aturan Statis Manual If-Else)
y_rule_based = ((df_ml["kehadiran_pct"] >= 80) & (df_ml["jam_belajar"] >= 12)).astype(int)
acc_rule = np.mean(y_true == y_rule_based)

# 3. Data-Driven Thinking (Supervised Learning & Unsupervised K-Means Clustering)
feats = df_ml[["jam_belajar", "kehadiran_pct", "skor_tugas"]].values
if HAS_SKLEARN:
    X_tr, X_te, y_tr, y_te = train_test_split(feats, y_true, test_size=0.25, random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_tr, y_tr)
    acc_ml = accuracy_score(y_te, clf_rf.predict(X_te))

    X_scaled = StandardScaler().fit_transform(feats)
    df_ml["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)
else:
    X_poly = np.column_stack([feats, (feats[:, 0] - 15) ** 2])
    mu, sd = X_poly.mean(axis=0), X_poly.std(axis=0) + 1e-8
    X_norm = (X_poly - mu) / sd
    w, b = np.zeros(X_norm.shape[1]), 0.0
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-(X_norm[:375] @ w + b)))
        w -= 0.15 * (X_norm[:375].T @ (p - y_true[:375])) / 375
        b -= 0.15 * np.mean(p - y_true[:375])
    pred_te = ((1.0 / (1.0 + np.exp(-(X_norm[375:] @ w + b)))) >= 0.5).astype(int)
    acc_ml = np.mean(pred_te == y_true[375:])

    X_s = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    centroids = X_s[np.array([10, 150, 300])]
    for _ in range(30):
        dists = np.linalg.norm(X_s[:, None, :] - centroids[None, :, :], axis=2)
        clusters = np.argmin(dists, axis=1)
        for k in range(3):
            if np.any(clusters == k):
                centroids[k] = X_s[clusters == k].mean(axis=0)
    df_ml["cluster"] = clusters

print(f"[1] Akurasi Pendekatan Rule-Based (If-Else Statis) : {acc_rule*100:.2f}%")
print(f"[2] Akurasi Pendekatan Data-Driven (Supervised ML) : {acc_ml*100:.2f}%\n")

print("[3] Profil Rata-Rata Cluster Mahasiswa (Unsupervised K-Means):")
print(df_ml.groupby("cluster")[["jam_belajar", "kehadiran_pct", "skor_tugas"]].mean().round(1).to_string())

# Visualisasi Mandiri Sesi 4
plt.figure(figsize=(8, 4.8))
palette = ["#2563EB", "#0D9488", "#F59E0B"]
for c in range(3):
    sub = df_ml[df_ml["cluster"] == c]
    plt.scatter(sub["jam_belajar"], sub["skor_tugas"], c=palette[c], label=f"Cluster {c}", alpha=0.75, s=40)
plt.title("Sesi 4: Pengelompokan Profil Belajar Mahasiswa dengan K-Means Clustering", fontweight="bold")
plt.xlabel("Jam Belajar Mandiri per Minggu")
plt.ylabel("Skor Tugas Rata-Rata")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("output_sesi_4_machine_learning.png", dpi=150)
plt.show()
plt.close()