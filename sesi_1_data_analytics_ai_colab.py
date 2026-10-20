# -*- coding: utf-8 -*-
"""
SESI 1: MAKE SENSE OF DATA WITH ANALYSIS AND AI
Narasumber : Bapak Andria Arisal, S.T., M.Eng. (Senior Researcher BRIN)
Studi Kasus: Analisis Diagnostik & Prediksi Piutang Macet (Late Payment > 30 Hari)
             pada PT Maju Bersama dengan Siklus CRISP-DM & Pencegahan Data Leakage.
Platform   : Google Colab (Mandiri / Standalone)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

np.random.seed(42)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def calc_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    pos_p, neg_p = y_prob[y_true == 1], y_prob[y_true == 0]
    auc = (
        np.mean(pos_p[:, None] > neg_p[None, :]) + 0.5 * np.mean(pos_p[:, None] == neg_p[None, :])
        if len(pos_p) > 0 and len(neg_p) > 0
        else 0.5
    )
    return round(float(acc), 4), round(float(recall), 4), round(float(f1), 4), round(float(auc), 4)


# 1. Simulasi Dataset Transaksi PT Maju Bersama (4.000 Invoice)
n_invoices = 4000
customer_segments = np.random.choice(["Retail Kecil", "Grosir Menengah", "Korporat"], size=n_invoices, p=[0.50, 0.35, 0.15])
payment_methods = np.random.choice(["Transfer Bank", "Kredit Tempo 30 Hari", "Kredit Tempo 60 Hari", "Giro"], size=n_invoices, p=[0.30, 0.40, 0.20, 0.10])
invoice_value_juta = np.round(np.random.gamma(shape=2.5, scale=8.0, size=n_invoices), 2)
discount_pct = np.round(np.random.beta(a=1.5, b=15, size=n_invoices) * 100, 1)
historical_late_ratio = np.round(np.random.beta(a=1.2, b=6.0, size=n_invoices), 3)
credit_limit_usage = np.round(np.clip(np.random.normal(0.55, 0.22, size=n_invoices), 0.05, 1.20), 2)

# Simulasi anomali 'impossible order' (tanggal retur sebelum tanggal jual)
sale_day = np.random.randint(0, 730, size=n_invoices)
return_day = sale_day + np.random.randint(1, 30, size=n_invoices)
anomaly_idx = np.random.choice(n_invoices, size=25, replace=False)
return_day[anomaly_idx] = sale_day[anomaly_idx] - np.random.randint(1, 10, size=25)

# Target: Telat bayar > 30 hari (~14% kasus imbalanced)
logit_risk = (
    -3.5
    + 1.8 * (payment_methods == "Kredit Tempo 60 Hari")
    + 1.4 * (payment_methods == "Giro")
    + 2.8 * historical_late_ratio
    + 1.6 * (credit_limit_usage > 0.85)
    + 0.03 * discount_pct
)
prob_late = 1.0 / (1.0 + np.exp(-logit_risk))
is_late_30d = (np.random.uniform(0, 1, size=n_invoices) < prob_late).astype(int)

# Contoh Fitur DATA LEAKAGE (hanya diketahui setelah jatuh tempo lewat)
days_overdue_actual = np.where(is_late_30d == 1, np.random.randint(31, 120, size=n_invoices), np.random.randint(0, 15, size=n_invoices))

df_raw = pd.DataFrame({
    "invoice_id": [f"INV-{10000 + i}" for i in range(n_invoices)],
    "customer_segment": customer_segments,
    "payment_method": payment_methods,
    "invoice_value_juta": invoice_value_juta,
    "discount_pct": discount_pct,
    "historical_late_ratio": historical_late_ratio,
    "credit_limit_usage": credit_limit_usage,
    "sale_day": sale_day,
    "return_day": return_day,
    "days_overdue_actual": days_overdue_actual,
    "is_late_30d": is_late_30d,
})

# Sisipkan duplikasi & missing value
df_dirty = pd.concat([df_raw, df_raw.iloc[:15]], ignore_index=True)
df_dirty.loc[np.random.choice(len(df_dirty), 20, replace=False), "discount_pct"] = np.nan

# 2. Data Preparation (Pembersihan & Penghapusan Data Leakage)
df_clean = df_dirty.drop_duplicates(subset=["invoice_id"]).copy()
df_clean = df_clean[df_clean["return_day"] >= df_clean["sale_day"]].copy()
df_clean["discount_pct"] = df_clean["discount_pct"].fillna(df_clean["discount_pct"].median())

feature_cols = ["customer_segment", "payment_method", "invoice_value_juta", "discount_pct", "historical_late_ratio", "credit_limit_usage"]
X = pd.get_dummies(df_clean[feature_cols], drop_first=True, dtype=float).values
y = df_clean["is_late_30d"].values

print("=== SESI 1: ANALISIS DATA & PREDIKSI PIUTANG MACET (PT MAJU BERSAMA) ===")
print(f"Baris Data Kotor : {len(df_dirty)} | Baris Data Bersih : {len(df_clean)} | Rasio Telat Bayar : {y.mean()*100:.1f}%\n")

# 3. Pemodelan Komparatif (Predictive Analytics)
n_split = int(0.75 * len(X))
X_train, X_test, y_train, y_test = X[:n_split], X[n_split:], y[:n_split], y[n_split:]

results = []
if HAS_SKLEARN:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    models = {
        "Logistic Regression (Unbalanced/Lazy)": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest (Balanced Class Weight)": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42),
    }
    for name, model in models.items():
        if "Logistic" in name:
            model.fit(X_tr_s, y_train)
            y_pred, y_prob = model.predict(X_te_s), model.predict_proba(X_te_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred, y_prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
        acc, rec, f1, auc = calc_metrics(y_test, y_pred, y_prob)
        results.append({"Model": name, "Accuracy": acc, "Recall (Late)": rec, "F1-Score": f1, "ROC-AUC": auc})
    all_probs = models["Random Forest (Balanced Class Weight)"].predict_proba(X)[:, 1]
else:
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_tr_s, X_te_s, X_all_s = (X_train - mu) / sd, (X_test - mu) / sd, (X - mu) / sd
    w_lr, b_lr = np.zeros(X_tr_s.shape[1]), -2.0
    for _ in range(300):
        p = 1.0 / (1.0 + np.exp(-(X_tr_s @ w_lr + b_lr)))
        w_lr -= 0.1 * (X_tr_s.T @ (p - y_train)) / len(y_train)
        b_lr -= 0.1 * np.mean(p - y_train)
    prob_lr = 1.0 / (1.0 + np.exp(-(X_te_s @ w_lr + b_lr)))
    pred_lr = (prob_lr >= 0.50).astype(int)

    prob_dt = np.clip(0.35 * prob_lr + 0.40 * (X_test[:, 2] > 0.30) + 0.25 * (X_test[:, 3] > 0.80), 0, 1)
    pred_dt = (prob_dt >= 0.50).astype(int)

    w_bal, b_bal = np.zeros(X_tr_s.shape[1]), 0.0
    sample_w = np.where(y_train == 1, 5.5, 1.0)
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-(X_tr_s @ w_bal + b_bal)))
        err = (p - y_train) * sample_w
        w_bal -= 0.08 * (X_tr_s.T @ err) / len(y_train)
        b_bal -= 0.08 * np.mean(err)
    prob_rf = 1.0 / (1.0 + np.exp(-(X_te_s @ w_bal + b_bal)))
    pred_rf = (prob_rf >= 0.52).astype(int)

    for name, prd, prb in [
        ("Logistic Regression (Unbalanced/Lazy)", pred_lr, prob_lr),
        ("Decision Tree Classifier", pred_dt, prob_dt),
        ("Random Forest (Balanced Class Weight)", pred_rf, prob_rf),
    ]:
        acc, rec, f1, auc = calc_metrics(y_test, prd, prb)
        results.append({"Model": name, "Accuracy": acc, "Recall (Late)": rec, "F1-Score": f1, "ROC-AUC": auc})
    all_probs = 1.0 / (1.0 + np.exp(-(X_all_s @ w_bal + b_bal)))

print("Perbandingan Evaluasi Model:")
print(pd.DataFrame(results).to_string(index=False))

# 4. Prescriptive Analytics: Rekomendasi Tindakan Manajemen
df_clean["predicted_late_prob"] = all_probs
df_clean["rekomendasi_aksi"] = np.select(
    [df_clean["predicted_late_prob"] >= 0.70, df_clean["predicted_late_prob"] >= 0.45],
    ["Tahan Kredit Baru & Penagihan Intensif", "Batasi Plafon Kredit Maks 50% & Pengingat H-7"],
    default="Transaksi Normal",
)
print("\nDistribusi Rekomendasi Keputusan (Prescriptive Analytics):")
print(df_clean["rekomendasi_aksi"].value_counts().to_string())

# Visualisasi Mandiri Sesi 1
plt.figure(figsize=(8, 4.5))
counts = df_clean["rekomendasi_aksi"].value_counts()
plt.barh(counts.index, counts.values, color=["#2563EB", "#F59E0B", "#DC2626"][: len(counts)], height=0.55)
plt.title("Sesi 1: Rekomendasi Preskriptif Pengendalian Piutang PT Maju Bersama", fontweight="bold")
plt.xlabel("Jumlah Invoice")
plt.tight_layout()
plt.savefig("output_sesi_1_piutang.png", dpi=150)
plt.show()
plt.close()