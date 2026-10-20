# -*- coding: utf-8 -*-
"""
SESI 6: AUTOMATED DEFORESTATION ALERTING AND DRIVER ATTRIBUTION
Narasumber : Dr. Bart Slachter (Wageningen University & Research, Belanda)
Studi Kasus: Klasifikasi Penyebab Langsung Deforestasi Hutan Tropis Menggunakan
             Gabungan Citra Satelit Sentinel-1 (Radar SAR) & Sentinel-2 (Optik)
             serta Benchmarking ML Konvensional vs Deep Learning (CNN Spatial).
Platform   : Google Colab (Mandiri / Standalone)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

np.random.seed(42)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def calc_f1_per_class(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return np.array(f1s)


# 1. Simulasi Patch Citra Multi-Band Sentinel-1 (2 Band Radar: VV, VH) & Sentinel-2 (4 Band Optik: R, G, B, NIR)
class_names = [
    "Smallholder Agri",
    "Large-Scale Agri",
    "Selective Logging",
    "Mining",
    "Wildfire",
    "Flooding",
]
n_samples = 1500
probs = [0.32, 0.28, 0.14, 0.10, 0.08, 0.08]
labels = np.random.choice(len(class_names), size=n_samples, p=probs)
patches = np.zeros((n_samples, 8, 8, 6), dtype=np.float32)

for i, y_cls in enumerate(labels):
    radar_vv = np.random.normal(0.55, 0.07, (8, 8))
    radar_vh = np.random.normal(0.45, 0.07, (8, 8))
    opt_r = np.random.normal(0.20, 0.04, (8, 8))
    opt_g = np.random.normal(0.35, 0.04, (8, 8))
    opt_b = np.random.normal(0.18, 0.04, (8, 8))
    opt_nir = np.random.normal(0.75, 0.06, (8, 8))

    if y_cls == 0:  # Smallholder Agriculture: Bukaan lahan kecil 3x3 di tengah
        radar_vh[2:5, 2:5] -= 0.20
        opt_r[2:5, 2:5] += 0.25
        opt_nir[2:5, 2:5] -= 0.32
    elif y_cls == 1:  # Large-Scale Agriculture: Bukaan lahan blok luas 6x8
        radar_vv[:6, :] -= 0.25
        radar_vh[:6, :] -= 0.28
        opt_r[:6, :] += 0.35
        opt_nir[:6, :] -= 0.45
    elif y_cls == 2:  # Selective Logging: Pola koridor jalan sempit memanjang
        radar_vh[:, 3:5] -= 0.22
        opt_r[:, 3:5] += 0.24
        opt_nir[:, 3:5] -= 0.28
    elif y_cls == 3:  # Mining: Tanah terbuka sangat terang & kolam air tambang
        radar_vv[1:7, 1:7] -= 0.34
        opt_r[1:7, 1:7] += 0.48
        opt_g[1:7, 1:7] += 0.32
        opt_nir[1:7, 1:7] -= 0.52
    elif y_cls == 4:  # Wildfire: Kerusakan kanopi di tepi sungai (mirip karakter banjir)
        radar_vv[1:6, 1:6] -= 0.33 + np.random.normal(0, 0.05)
        radar_vh[1:6, 1:6] -= 0.29 + np.random.normal(0, 0.05)
        opt_r[1:6, 1:6] += 0.14
        opt_nir[1:6, 1:6] -= 0.48
    elif y_cls == 5:  # Flooding: Genangan air menurunkan hamburan balik radar VV/VH
        radar_vv[1:6, 1:6] -= 0.36 + np.random.normal(0, 0.05)
        radar_vh[1:6, 1:6] -= 0.31 + np.random.normal(0, 0.05)
        opt_r[1:6, 1:6] += 0.12
        opt_nir[1:6, 1:6] -= 0.46

    # Simulasi 35% citra optik Sentinel-2 tertutup awan tropis (sehingga model mengandalkan Sentinel-1 Radar)
    if np.random.rand() < 0.35:
        cloud = np.random.uniform(0.85, 0.98, (8, 8))
        opt_r, opt_g, opt_b, opt_nir = cloud, cloud, cloud, cloud

    patches[i] = np.stack([radar_vv, radar_vh, opt_r, opt_g, opt_b, opt_nir], axis=-1)

# 2. Ekstraksi Fitur Piksel Rata-Rata (ML Konvensional) vs Fitur Konvolusi Spasial 2D (CNN)
X_simple = patches.mean(axis=(1, 2))
feat_list = []
for i in range(n_samples):
    img = patches[i]
    sample_feats = []
    for b in range(6):
        band = img[:, :, b]
        grad_x = np.abs(np.diff(band, axis=1)).mean()
        grad_y = np.abs(np.diff(band, axis=0)).mean()
        center_3x3 = band[2:5, 2:5].mean()
        strip_vert = band[:, 3:5].mean()
        border_val = np.concatenate([band[0, :], band[-1, :], band[:, 0], band[:, -1]]).mean()
        sample_feats.extend([band.mean(), band.std(), center_3x3 - border_val, strip_vert - border_val, grad_x, grad_y, grad_x / (grad_y + 1e-5)])
    feat_list.append(sample_feats)
X_spatial = np.array(feat_list, dtype=np.float32)

# 3. Pelatihan & Benchmarking Model
n_tr = int(0.75 * n_samples)
y_tr, y_te = labels[:n_tr], labels[n_tr:]
n_classes = len(class_names)

if HAS_SKLEARN:
    rf_conv = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_conv.fit(X_simple[:n_tr], y_tr)
    pred_rf = rf_conv.predict(X_simple[n_tr:])

    scaler_dl = StandardScaler()
    X_tr_dl = scaler_dl.fit_transform(X_spatial[:n_tr])
    X_te_dl = scaler_dl.transform(X_spatial[n_tr:])
    cnn_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)
    cnn_mlp.fit(X_tr_dl, y_tr)
    pred_cnn = cnn_mlp.predict(X_te_dl)
else:
    mu_s, sd_s = X_simple[:n_tr].mean(axis=0), X_simple[:n_tr].std(axis=0) + 1e-8
    Xs_tr, Xs_te = (X_simple[:n_tr] - mu_s) / sd_s, (X_simple[n_tr:] - mu_s) / sd_s
    centroids_rf = np.array([Xs_tr[y_tr == c].mean(axis=0) for c in range(n_classes)])
    pred_rf = np.argmin(np.linalg.norm(Xs_te[:, None, :] - centroids_rf[None, :, :], axis=2), axis=1)

    mu_d, sd_d = X_spatial[:n_tr].mean(axis=0), X_spatial[:n_tr].std(axis=0) + 1e-8
    Xd_tr, Xd_te = (X_spatial[:n_tr] - mu_d) / sd_d, (X_spatial[n_tr:] - mu_d) / sd_d
    y_onehot = np.eye(n_classes)[y_tr]
    w1, b1 = np.random.randn(Xd_tr.shape[1], 64) * 0.15, np.zeros(64)
    w2, b2 = np.random.randn(64, n_classes) * 0.15, np.zeros(n_classes)
    for _ in range(450):
        h1 = np.maximum(0, Xd_tr @ w1 + b1)
        logits = h1 @ w2 + b2
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs_out = exp_l / exp_l.sum(axis=1, keepdims=True)
        dlogits = (probs_out - y_onehot) / len(y_tr)
        dw2, db2 = h1.T @ dlogits, dlogits.sum(axis=0)
        dh1 = (dlogits @ w2.T) * (h1 > 0)
        dw1, db1 = Xd_tr.T @ dh1, dh1.sum(axis=0)
        w2 -= 0.25 * dw2
        b2 -= 0.25 * db2
        w1 -= 0.25 * dw1
        b1 -= 0.25 * db1
    pred_cnn = np.argmax(np.maximum(0, Xd_te @ w1 + b1) @ w2 + b2, axis=1)

acc_rf, f1_rf = np.mean(pred_rf == y_te), calc_f1_per_class(y_te, pred_rf, n_classes)
acc_cnn, f1_cnn = np.mean(pred_cnn == y_te), calc_f1_per_class(y_te, pred_cnn, n_classes)

print("=== SESI 6: KLASIFIKASI PENYEBAB DEFORESTASI (SENTINEL-1/2 & DEEP LEARNING) ===")
print(f"[1] ML Konvensional (Piksel Rata-Rata) : Overall Acc = {acc_rf*100:.2f}% | Macro F1-Score = {f1_rf.mean()*100:.2f}%")
print(f"[2] Deep Learning (Konteks Spasial CNN): Overall Acc = {acc_cnn*100:.2f}% | Macro F1-Score = {f1_cnn.mean()*100:.2f}%\n")

df_metrics = pd.DataFrame({
    "Kelas Penyebab Deforestasi": class_names,
    "F1 ML Konvensional": np.round(f1_rf, 3),
    "F1 Deep Learning (CNN)": np.round(f1_cnn, 3),
})
print(df_metrics.to_string(index=False))

# Visualisasi Mandiri Sesi 6
plt.figure(figsize=(8.5, 4.8))
x_pos = np.arange(n_classes)
plt.bar(x_pos - 0.18, f1_rf, width=0.35, color="#94A3B8", label="ML Konvensional (Tanpa Konteks Spasial)")
plt.bar(x_pos + 0.18, f1_cnn, width=0.35, color="#1D4ED8", label="Deep Learning CNN (Dengan Konteks Spasial)")
plt.xticks(x_pos, class_names, rotation=15, ha="right")
plt.ylim(0, 1.08)
plt.ylabel("F1-Score")
plt.title("Sesi 6: Perbandingan F1-Score Klasifikasi Penyebab Deforestasi (Sentinel-1 & Sentinel-2)", fontweight="bold")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("output_sesi_6_deforestasi_cnn.png", dpi=150)
plt.show()
plt.close()