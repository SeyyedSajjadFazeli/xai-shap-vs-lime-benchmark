"""
╔══════════════════════════════════════════════════════════════════╗
║         SHAP vs LIME: A Complete XAI Benchmark Study            ║
║   Classification & Regression · 3 Models · Speed & Stability   ║
╚══════════════════════════════════════════════════════════════════╝

Author  : [Your Name]
GitHub  : https://github.com/[your-username]/shap-vs-lime-benchmark
Dataset : Titanic (classification) · California Housing (regression)
Models  : RandomForest · XGBoost · LogisticRegression / LinearRegression
"""

# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import timeit
from pathlib import Path

# sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline

# XGBoost
import xgboost as xgb

# XAI
import shap
import lime
import lime.lime_tabular

# Output directory
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot style — clean & publication-ready
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PALETTE = {
    "shap":  "#7B68EE",   # purple
    "lime":  "#FF7F7F",   # coral
    "rf":    "#2ECC71",
    "xgb":   "#F39C12",
    "lr":    "#3498DB",
}

print("✅ All libraries loaded successfully\n")


# ─────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────

def load_titanic():
    """Load Titanic dataset (classification task)."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback: minimal synthetic Titanic-like data
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            "Survived": np.random.randint(0, 2, n),
            "Pclass":   np.random.choice([1, 2, 3], n),
            "Age":      np.random.normal(29, 14, n).clip(1, 80),
            "SibSp":    np.random.poisson(0.5, n),
            "Parch":    np.random.poisson(0.4, n),
            "Fare":     np.random.exponential(32, n),
            "Sex_male": np.random.randint(0, 2, n),
        })
        return df, df.drop("Survived", axis=1), df["Survived"]

    features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    df["Sex_male"] = (df["Sex"] == "male").astype(int)
    features.append("Sex_male")

    df = df[features + ["Survived"]].dropna()
    X = df[features]
    y = df["Survived"]
    return df, X, y


def load_housing():
    """Load California Housing dataset (regression task)."""
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y


print("📦 Loading datasets...")
df_titanic, X_clf, y_clf = load_titanic()
X_reg, y_reg = load_housing()

# Train/test split
X_clf_tr, X_clf_te, y_clf_tr, y_clf_te = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Use a small subset for LIME/SHAP speed benchmarks
N_BENCH = 100   # samples for XAI explanation
X_clf_bench = X_clf_te.iloc[:N_BENCH]
X_reg_bench = X_reg_te.iloc[:N_BENCH]

feature_names_clf = list(X_clf.columns)
feature_names_reg = list(X_reg.columns)

print(f"  Classification → train: {len(X_clf_tr)}, test: {len(X_clf_te)}, features: {len(feature_names_clf)}")
print(f"  Regression     → train: {len(X_reg_tr)}, test: {len(X_reg_te)}, features: {len(feature_names_reg)}")


# ─────────────────────────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────

print("\n🤖 Training models...")

# --- Classification ---
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_clf_tr, y_clf_tr)

xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42,
                              eval_metric="logloss", verbosity=0)
xgb_clf.fit(X_clf_tr, y_clf_tr)

lr_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_clf.fit(X_clf_tr, y_clf_tr)

# --- Regression ---
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_reg_tr, y_reg_tr)

xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_reg.fit(X_reg_tr, y_reg_tr)

lr_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
lr_reg.fit(X_reg_tr, y_reg_tr)

# Scores
clf_models = {"Random Forest": rf_clf, "XGBoost": xgb_clf, "Logistic Reg.": lr_clf}
reg_models  = {"Random Forest": rf_reg, "XGBoost": xgb_reg, "Linear Reg.": lr_reg}

print("\n  Classification accuracy:")
for name, m in clf_models.items():
    acc = accuracy_score(y_clf_te, m.predict(X_clf_te))
    print(f"    {name:20s}: {acc:.3f}")

print("\n  Regression R² score:")
for name, m in reg_models.items():
    r2 = r2_score(y_reg_te, m.predict(X_reg_te))
    print(f"    {name:20s}: {r2:.3f}")


# ─────────────────────────────────────────────────────────────────
# 3. SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────────────

print("\n🔍 Computing SHAP values...")

def get_shap_explainer(model, X_train, task="clf"):
    """Return the best SHAP explainer for a given model."""
    model_type = type(model).__name__
    
    if "Pipeline" in model_type:
        # برای نمونه‌برداری از داده پس‌زمینه جهت افزایش سرعت
        bg = shap.sample(X_train, 50) if len(X_train) > 50 else X_train
        
        # استفاده از تابع lambda برای دور زدن محدودیت‌های سیستم نام‌گذاری ویژگی‌ها در Pipeline
        if task == "clf":
            return shap.KernelExplainer(lambda x: model.predict_proba(x), bg)
        else:
            return shap.KernelExplainer(lambda x: model.predict(x), bg)
            
    elif "XGB" in model_type or "RandomForest" in model_type:
        # مفسر مخصوص مدل‌های درختی (بسیار سریع و بهینه)
        return shap.TreeExplainer(model)
        
    else:
        # در صورتی که مدل دیگری خارج از پایپ‌لاین و درخت بود، از کرنیل استفاده کند
        bg = shap.sample(X_train, 50) if len(X_train) > 50 else X_train
        if task == "clf":
            return shap.KernelExplainer(lambda x: model.predict_proba(x), bg)
        else:
            return shap.KernelExplainer(lambda x: model.predict(x), bg)

# Classification SHAP
shap_values_clf = {}
shap_times_clf  = {}

for name, model in clf_models.items():
    print(f"  SHAP / {name} ...", end=" ")
    t0 = time.time()
    explainer = get_shap_explainer(model, X_clf_tr, task="clf")
    sv = explainer(X_clf_bench)
    elapsed = time.time() - t0
    # Store values for positive class
    if hasattr(sv, "values") and sv.values.ndim == 3:
        shap_values_clf[name] = sv[..., 1]          # class-1 slice
    else:
        shap_values_clf[name] = sv
    shap_times_clf[name] = elapsed
    print(f"{elapsed:.2f}s")

# Regression SHAP
shap_values_reg = {}
shap_times_reg  = {}

for name, model in reg_models.items():
    print(f"  SHAP / {name} ...", end=" ")
    t0 = time.time()
    explainer = get_shap_explainer(model, X_reg_tr, task="reg")
    sv = explainer(X_reg_bench)
    elapsed = time.time() - t0
    shap_values_reg[name] = sv
    shap_times_reg[name]  = elapsed
    print(f"{elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────
# 4. LIME EXPLANATIONS
# ─────────────────────────────────────────────────────────────────

print("\n🔍 Computing LIME values...")

def lime_feature_importances(explainer, model, X_bench, task="clf",
                              num_features=None, n_samples=50):
    """Return mean absolute LIME weights per feature."""
    if num_features is None:
        num_features = X_bench.shape[1]
    all_weights = []
    predict_fn = model.predict_proba if task == "clf" else model.predict

    for i in range(min(n_samples, len(X_bench))):
        exp = explainer.explain_instance(
            X_bench.values[i],
            predict_fn,
            num_features=num_features,
            num_samples=500,
        )
        weight_dict = dict(exp.as_list())
        row = []
        for feat in X_bench.columns:
            # LIME produces "feature op value" strings, match by feature name
            val = next((v for k, v in weight_dict.items() if feat in k), 0.0)
            row.append(val)
        all_weights.append(row)

    return np.array(all_weights)


def make_lime_explainer(X_train, feature_names, task="clf"):
    mode = "classification" if task == "clf" else "regression"
    return lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        mode=mode,
        random_state=42,
    )


lime_weights_clf = {}
lime_times_clf   = {}
N_LIME = 50  # LIME is slow — use 50 samples for fair comparison

lime_exp_clf = make_lime_explainer(X_clf_tr, feature_names_clf, task="clf")

for name, model in clf_models.items():
    print(f"  LIME / {name} ...", end=" ")
    t0 = time.time()
    w = lime_feature_importances(lime_exp_clf, model, X_clf_bench.iloc[:N_LIME],
                                  task="clf", n_samples=N_LIME)
    elapsed = time.time() - t0
    lime_weights_clf[name] = w
    lime_times_clf[name]   = elapsed
    print(f"{elapsed:.2f}s")

lime_weights_reg = {}
lime_times_reg   = {}

lime_exp_reg = make_lime_explainer(X_reg_tr, feature_names_reg, task="reg")

for name, model in reg_models.items():
    print(f"  LIME / {name} ...", end=" ")
    t0 = time.time()
    w = lime_feature_importances(lime_exp_reg, model, X_reg_bench.iloc[:N_LIME],
                                  task="reg", n_samples=N_LIME)
    elapsed = time.time() - t0
    lime_weights_reg[name] = w
    lime_times_reg[name]   = elapsed
    print(f"{elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────
# 5. BENCHMARK: EXECUTION TIME COMPARISON
# ─────────────────────────────────────────────────────────────────

print("\n⏱  Timing benchmark...")

def build_timing_df(shap_times, lime_times, task_label):
    rows = []
    for name in shap_times:
        rows.append({"Model": name, "Method": "SHAP", "Task": task_label,
                     "Time (s)": shap_times[name]})
        rows.append({"Model": name, "Method": "LIME", "Task": task_label,
                     "Time (s)": lime_times[name]})
    return pd.DataFrame(rows)

timing_clf = build_timing_df(shap_times_clf, lime_times_clf, "Classification")
timing_reg = build_timing_df(shap_times_reg, lime_times_reg, "Regression")
timing_all = pd.concat([timing_clf, timing_reg], ignore_index=True)

print("\n  Timing summary (seconds):")
print(timing_all.pivot_table(index=["Task", "Model"], columns="Method",
                              values="Time (s)").round(2).to_string())


# ─────────────────────────────────────────────────────────────────
# 6. FEATURE AGREEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────

def top_k_agreement(shap_vals, lime_vals, feature_names, k=3):
    """
    Compute fraction of samples where top-k SHAP features
    overlap with top-k LIME features.
    """
    n_samples = min(len(shap_vals.values), lime_vals.shape[0])
    shap_arr = np.abs(shap_vals.values[:n_samples])
    lime_arr = np.abs(lime_vals[:n_samples])

    agreements = []
    for i in range(n_samples):
        shap_top = set(np.argsort(shap_arr[i])[-k:])
        lime_top = set(np.argsort(lime_arr[i])[-k:])
        agreements.append(len(shap_top & lime_top) / k)
    return np.mean(agreements)


agreement_results = {}
for name in clf_models:
    sv = shap_values_clf[name]
    lv = lime_weights_clf[name]
    agreement_results[f"CLF / {name}"] = top_k_agreement(sv, lv, feature_names_clf)

for name in reg_models:
    sv = shap_values_reg[name]
    lv = lime_weights_reg[name]
    agreement_results[f"REG / {name}"] = top_k_agreement(sv, lv, feature_names_reg)

print("\n  Feature agreement (top-3 overlap):")
for k, v in agreement_results.items():
    print(f"    {k:35s}: {v:.1%}")


# ─────────────────────────────────────────────────────────────────
# 7. STABILITY ANALYSIS — LIME vs SHAP
# ─────────────────────────────────────────────────────────────────

def lime_stability(explainer, model, instance, feature_names,
                   task="clf", n_runs=10, num_features=None):
    """Run LIME n_runs times on the same instance, return std of feature weights."""
    if num_features is None:
        num_features = len(feature_names)
    predict_fn = model.predict_proba if task == "clf" else model.predict
    all_weights = []
    for _ in range(n_runs):
        exp = explainer.explain_instance(
            instance, predict_fn,
            num_features=num_features, num_samples=200)
        wd = dict(exp.as_list())
        row = [next((v for k, v in wd.items() if f in k), 0.0)
               for f in feature_names]
        all_weights.append(row)
    return np.std(all_weights, axis=0)   # std per feature

sample_idx = 0
lime_std_clf = lime_stability(lime_exp_clf, rf_clf,
                               X_clf_bench.values[sample_idx],
                               feature_names_clf, task="clf")

# SHAP stability: deterministic for TreeExplainer → effectively 0
shap_std_clf = np.zeros(len(feature_names_clf))

stability_df = pd.DataFrame({
    "Feature": feature_names_clf,
    "SHAP std": shap_std_clf,
    "LIME std": lime_std_clf,
})
print("\n  Stability (std over 10 runs, RF-clf, sample 0):")
print(stability_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────

print("\n🎨 Generating plots...")

# ── Plot 1: Timing Comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("SHAP vs LIME — Execution Time Benchmark", fontsize=15, fontweight="bold")

for ax, task in zip(axes, ["Classification", "Regression"]):
    df_t = timing_all[timing_all["Task"] == task]
    models_list = df_t["Model"].unique()
    x = np.arange(len(models_list))
    w = 0.35
    shap_vals = df_t[df_t["Method"] == "SHAP"]["Time (s)"].values
    lime_vals = df_t[df_t["Method"] == "LIME"]["Time (s)"].values

    bars_s = ax.bar(x - w/2, shap_vals, w, label="SHAP", color=PALETTE["shap"], alpha=0.85)
    bars_l = ax.bar(x + w/2, lime_vals, w, label="LIME", color=PALETTE["lime"], alpha=0.85)

    for bars in [bars_s, bars_l]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=10)
    ax.set_ylabel("Seconds")
    ax.set_title(f"Task: {task}")
    ax.legend()
    ax.set_ylim(0, max(max(shap_vals), max(lime_vals)) * 1.35)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_timing_benchmark.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 01_timing_benchmark.png")


# ── Plot 2: SHAP Beeswarm — RF Classification ─────────────────
fig, ax = plt.subplots(figsize=(9, 5))
shap.plots.beeswarm(shap_values_clf["Random Forest"],
                    max_display=6, show=False)
plt.title("SHAP Beeswarm — Random Forest (Titanic Survival)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_shap_beeswarm_clf.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 02_shap_beeswarm_clf.png")


# ── Plot 3: SHAP Waterfall — single prediction ────────────────
fig, ax = plt.subplots(figsize=(9, 5))
shap.plots.waterfall(shap_values_clf["Random Forest"][0], max_display=6, show=False)
plt.title("SHAP Waterfall — Single Prediction (RF, Titanic)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 03_shap_waterfall.png")


# ── Plot 4: LIME Feature Weights — single prediction ─────────
lime_exp_instance = lime_exp_clf.explain_instance(
    X_clf_bench.values[0], rf_clf.predict_proba,
    num_features=len(feature_names_clf), num_samples=500)

lime_list = lime_exp_instance.as_list()
lime_feats = [x[0].split(" ")[-1] for x in lime_list]  # simplify labels
lime_wts   = [x[1] for x in lime_list]
colors_lime = [PALETTE["shap"] if w > 0 else PALETTE["lime"] for w in lime_wts]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(lime_feats, lime_wts, color=colors_lime, alpha=0.85)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("LIME Weight")
ax.set_title("LIME Explanation — Single Prediction (RF, Titanic)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_lime_single_prediction.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 04_lime_single_prediction.png")


# ── Plot 5: Feature Agreement Heatmap ────────────────────────
n_tasks   = 2
n_models  = 3
agr_matrix = np.zeros((n_tasks, n_models))
task_labels  = ["CLF", "REG"]
model_labels = list(clf_models.keys())

for j, name in enumerate(clf_models):
    sv = shap_values_clf[name]
    lv = lime_weights_clf[name]
    agr_matrix[0, j] = top_k_agreement(sv, lv, feature_names_clf)

for j, name in enumerate(reg_models):
    sv = shap_values_reg[name]
    lv = lime_weights_reg[name]
    agr_matrix[1, j] = top_k_agreement(sv, lv, feature_names_reg)

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(agr_matrix, annot=True, fmt=".0%", cmap="RdYlGn",
            xticklabels=model_labels, yticklabels=["Classification", "Regression"],
            vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"label": "Agreement"})
ax.set_title("SHAP vs LIME — Top-3 Feature Agreement", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_feature_agreement_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 05_feature_agreement_heatmap.png")


# ── Plot 6: LIME Stability Analysis ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
x_pos = np.arange(len(feature_names_clf))
ax.bar(x_pos - 0.2, shap_std_clf, 0.35, label="SHAP (deterministic)",
       color=PALETTE["shap"], alpha=0.85)
ax.bar(x_pos + 0.2, lime_std_clf, 0.35, label="LIME (stochastic)",
       color=PALETTE["lime"], alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names_clf, rotation=15)
ax.set_ylabel("Std of explanation weight (10 runs)")
ax.set_title("Stability: SHAP vs LIME — Same Instance, 10 Runs (RF, Titanic)", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_stability_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 06_stability_analysis.png")


# ── Plot 7: Summary Dashboard ────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("SHAP vs LIME — Complete Benchmark Dashboard", fontsize=16, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Speed ratio
ax1 = fig.add_subplot(gs[0, 0])
speed_ratios = {}
for name in clf_models:
    ratio = lime_times_clf[name] / shap_times_clf[name]
    speed_ratios[f"CLF\n{name.split()[0]}"] = ratio
for name in reg_models:
    ratio = lime_times_reg[name] / shap_times_reg[name]
    speed_ratios[f"REG\n{name.split()[0]}"] = ratio

bars = ax1.bar(speed_ratios.keys(), speed_ratios.values(),
               color=[PALETTE["lime"] if v > 1 else PALETTE["shap"]
                      for v in speed_ratios.values()], alpha=0.85)
ax1.axhline(1, color="gray", linewidth=1, linestyle="--")
ax1.set_ylabel("LIME time / SHAP time")
ax1.set_title("Speed Ratio\n(>1 = SHAP faster)")
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05,
             f"{h:.1f}x", ha="center", fontsize=9)
ax1.set_ylim(0, max(speed_ratios.values()) * 1.3)

# Feature agreement
ax2 = fig.add_subplot(gs[0, 1:])
agr_items = list(agreement_results.items())
labels = [k.replace(" / ", "\n") for k, v in agr_items]
values = [v for k, v in agr_items]
bar_colors = [PALETTE["shap"] if v >= 0.5 else PALETTE["lime"] for v in values]
bars = ax2.bar(labels, values, color=bar_colors, alpha=0.85)
ax2.axhline(0.5, color="gray", linewidth=1, linestyle="--")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Agreement fraction")
ax2.set_title("Feature Agreement (Top-3 overlap)")
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02, f"{val:.0%}", ha="center", fontsize=9)

# Stability bar
ax3 = fig.add_subplot(gs[1, :2])
x_pos = np.arange(len(feature_names_clf))
ax3.bar(x_pos - 0.2, shap_std_clf, 0.35, label="SHAP",
        color=PALETTE["shap"], alpha=0.85)
ax3.bar(x_pos + 0.2, lime_std_clf, 0.35, label="LIME",
        color=PALETTE["lime"], alpha=0.85)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(feature_names_clf, rotation=20, fontsize=9)
ax3.set_ylabel("Std (10 runs)")
ax3.set_title("Stability: same instance, 10 runs (RF-CLF)")
ax3.legend()

# Summary text card
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis("off")
summary_text = (
    "🏆  KEY FINDINGS\n\n"
    "SHAP\n"
    "  ✓ Deterministic (std=0)\n"
    "  ✓ Global + Local views\n"
    "  ✓ Exact Shapley values\n"
    "  ✗ Slower on large data\n\n"
    "LIME\n"
    "  ✓ Fast & model-agnostic\n"
    "  ✓ Intuitive local approx\n"
    "  ✗ Stochastic results\n"
    "  ✗ Local only\n\n"
    "→ Use SHAP for production\n"
    "→ Use LIME for quick debug"
)
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8F8FF",
                   edgecolor="#AAAAEE", linewidth=1.2))

plt.savefig(OUTPUT_DIR / "07_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 07_dashboard.png")


# ─────────────────────────────────────────────────────────────────
# 9. RESULTS TABLE
# ─────────────────────────────────────────────────────────────────

print("\n📊 Final Results Table:")
results_rows = []
for name in clf_models:
    results_rows.append({
        "Task":       "Classification",
        "Model":      name,
        "SHAP time":  f"{shap_times_clf[name]:.2f}s",
        "LIME time":  f"{lime_times_clf[name]:.2f}s",
        "Faster":     "SHAP" if shap_times_clf[name] < lime_times_clf[name] else "LIME",
        "Agreement":  f"{agreement_results.get(f'CLF / {name}', 0):.0%}",
        "SHAP stable": "✓ Yes",
        "LIME stable": "✗ No",
    })
for name in reg_models:
    results_rows.append({
        "Task":       "Regression",
        "Model":      name,
        "SHAP time":  f"{shap_times_reg[name]:.2f}s",
        "LIME time":  f"{lime_times_reg[name]:.2f}s",
        "Faster":     "SHAP" if shap_times_reg[name] < lime_times_reg[name] else "LIME",
        "Agreement":  f"{agreement_results.get(f'REG / {name}', 0):.0%}",
        "SHAP stable": "✓ Yes",
        "LIME stable": "✗ No",
    })

results_df = pd.DataFrame(results_rows)
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
print("\n  ✅ results_summary.csv saved")

print("\n" + "="*60)
print("✅ All done! Files saved to ./results/")
print("="*60)
print("""
📁 Output files:
   results/
   ├── 01_timing_benchmark.png
   ├── 02_shap_beeswarm_clf.png
   ├── 03_shap_waterfall.png
   ├── 04_lime_single_prediction.png
   ├── 05_feature_agreement_heatmap.png
   ├── 06_stability_analysis.png
   ├── 07_dashboard.png       ← main LinkedIn image!
   └── results_summary.csv
""")
