
# ─── 0. Imports ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, LeaveOneOut, train_test_split
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# ─── 1. Configuration ────────────────────────────────────────────────────────
TRAIN_PATH  = "train.csv"
TEST_PATH   = "test.csv"
SAMPLE_PATH = "sample_submission.csv"
OUT_PATH    = "submission.csv"
PLOTS_DIR   = "plots"
SEED        = 42
N_FOLDS     = 5
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── 2. Load Data ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  LOADING DATA")
print("="*65)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sub   = pd.read_csv(SAMPLE_PATH)

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"\nTrain columns:\n{list(train.columns)}")
print(f"\nFirst 3 rows:\n{train.head(3)}")
print(f"\nMissing values in train:\n{train.isnull().sum()[train.isnull().sum()>0]}")
print(f"\nMissing values in test:\n{test.isnull().sum()[test.isnull().sum()>0]}")

# ─── 3. Auto-detect target and ID column ─────────────────────────────────────
# The target column is typically the one NOT in test but IS in sample_submission
TARGET = [c for c in train.columns if c not in test.columns
          and c not in ("id", "Id", "ID")][0]
ID_COL  = sub.columns[0]          # first column of sample_submission is the id
print(f"\n  → Target column : '{TARGET}'")
print(f"  → ID column     : '{ID_COL}'")

# ─── 4. Feature Engineering ──────────────────────────────────────────────────
print("\n" + "="*65)
print("  FEATURE ENGINEERING")
print("="*65)

all_data = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

# 4a. Encode categoricals
label_encoders = {}
for col in all_data.select_dtypes(include=["object", "category"]).columns:
    if col == ID_COL:
        continue
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))
    label_encoders[col] = le
    print(f"  Label-encoded : {col}")

# 4b. Numeric imputation (median)
for col in all_data.select_dtypes(include=[np.number]).columns:
    if all_data[col].isnull().any():
        median = all_data[col].median()
        all_data[col].fillna(median, inplace=True)
        print(f"  Imputed (median={median:.2f}) : {col}")

# 4c. Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(train[TARGET].astype(str))
n_classes = len(np.unique(y))
print(f"\n  Classes : {le_target.classes_}  (n={n_classes})")

# 4d. Split back
feature_cols = [c for c in all_data.columns if c != ID_COL]
X      = all_data.iloc[:len(train)][feature_cols].values
X_test = all_data.iloc[len(train):][feature_cols].values

# Scaled version (for LR, NB, KMeans)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_test_sc = scaler.transform(X_test)

print(f"\n  X shape     : {X.shape}")
print(f"  X_test shape: {X_test.shape}")

# ─── 5. K-Fold Cross-Validation Helper ───────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def kfold_cv(model, X_, y_, label=""):
    scores = cross_val_score(model, X_, y_,
                             cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  {label:30s}  CV Acc: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()

# ─── 6. Baseline Models ───────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  BASELINE MODELS  ({N_FOLDS}-Fold CV)")
print("="*65)

results = {}

# 6a. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=SEED)
results["Logistic Regression"] = kfold_cv(lr, X_sc, y, "Logistic Regression")

# 6b. Decision Tree
dt_cv_best, dt_best_depth = 0, 1
for depth in [3, 5, 8, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    acc = kfold_cv(dt, X, y, f"Decision Tree (depth={depth})")
    if acc > dt_cv_best:
        dt_cv_best = acc
        dt_best_depth = depth
results["Decision Tree"] = dt_cv_best

# 6c. Naive Bayes
nb = GaussianNB()
results["Naive Bayes"] = kfold_cv(nb, X_sc, y, "Naive Bayes")

# 6d. K-Means (as feature augmentation)
print("\n  [K-Means] Fitting unsupervised clusters as extra feature …")
km = KMeans(n_clusters=min(10, n_classes * 3), random_state=SEED, n_init=10)
km.fit(X_sc)
km_feat_train = km.predict(X_sc).reshape(-1, 1)
km_feat_test  = km.predict(X_test_sc).reshape(-1, 1)
X_km      = np.hstack([X,    km_feat_train])
X_test_km = np.hstack([X_test, km_feat_test])
dt_km = DecisionTreeClassifier(max_depth=dt_best_depth, random_state=SEED)
results["Decision Tree + KMeans"] = kfold_cv(dt_km, X_km, y,
                                              "Decision Tree + KMeans feat")

# 6e. Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
results["Random Forest"] = kfold_cv(rf, X, y, "Random Forest")

# ─── 7. LOOCV on a small stratified sample ───────────────────────────────────
print("\n" + "="*65)
print("  LOOCV  (on 200-sample stratified subset)")
print("="*65)
_, X_small, _, y_small = train_test_split(
    X_sc, y, test_size=200, stratify=y, random_state=SEED)
loo = LeaveOneOut()
for name, clf in [("Logistic Regression", LogisticRegression(max_iter=500, random_state=SEED)),
                  ("Naive Bayes",         GaussianNB()),
                  ("Decision Tree",       DecisionTreeClassifier(max_depth=dt_best_depth,
                                                                  random_state=SEED))]:
    loo_scores = cross_val_score(clf, X_small, y_small, cv=loo, scoring="accuracy")
    print(f"  {name:30s}  LOOCV Acc: {loo_scores.mean():.4f}")

# ─── 8. Final Ensemble: LightGBM + XGBoost + CatBoost ───────────────────────
print("\n" + "="*65)
print("  ENSEMBLE: LightGBM × XGBoost × CatBoost")
print("="*65)

lgb_params = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    max_depth=-1, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
    n_jobs=-1, verbose=-1
)
xgb_params = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, use_label_encoder=False,
    eval_metric="mlogloss", random_state=SEED, verbosity=0, n_jobs=-1,
    early_stopping_rounds=50
)
cat_params = dict(
    iterations=1000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, random_seed=SEED, verbose=0,
    eval_metric="Accuracy"
)

oof_lgb = np.zeros((len(y), n_classes))
oof_xgb = np.zeros((len(y), n_classes))
oof_cat = np.zeros((len(y), n_classes))

test_lgb = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_cat = np.zeros((len(X_test), n_classes))

fold_accs_lgb, fold_accs_xgb, fold_accs_cat = [], [], []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    Xtr, Xvl = X[trn_idx], X[val_idx]
    ytr, yvl = y[trn_idx], y[val_idx]
    print(f"\n  ── Fold {fold}/{N_FOLDS} ──")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(Xtr, ytr,
                  eval_set=[(Xvl, yvl)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
    oof_lgb[val_idx] = lgb_model.predict_proba(Xvl)
    test_lgb         += lgb_model.predict_proba(X_test) / N_FOLDS
    a = accuracy_score(yvl, oof_lgb[val_idx].argmax(1))
    fold_accs_lgb.append(a)
    print(f"    LGB  acc={a:.4f}")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        Xtr,
        ytr,
        eval_set=[(Xvl, yvl)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(Xvl)
    test_xgb         += xgb_model.predict_proba(X_test) / N_FOLDS
    a = accuracy_score(yvl, oof_xgb[val_idx].argmax(1))
    fold_accs_xgb.append(a)
    print(f"    XGB  acc={a:.4f}")

    # CatBoost
    cat_model = cb.CatBoostClassifier(**cat_params)
    cat_model.fit(Xtr, ytr,
                  eval_set=(Xvl, yvl),
                  early_stopping_rounds=50,
                  verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(Xvl)
    test_cat         += cat_model.predict_proba(X_test) / N_FOLDS
    a = accuracy_score(yvl, oof_cat[val_idx].argmax(1))
    fold_accs_cat.append(a)
    print(f"    CAT  acc={a:.4f}")

print(f"\n  LGB  OOF mean acc : {np.mean(fold_accs_lgb):.4f}")
print(f"  XGB  OOF mean acc : {np.mean(fold_accs_xgb):.4f}")
print(f"  CAT  OOF mean acc : {np.mean(fold_accs_cat):.4f}")

# Soft-vote ensemble
oof_ens  = (oof_lgb  + oof_xgb  + oof_cat)  / 3.0
test_ens = (test_lgb + test_xgb + test_cat) / 3.0

ens_acc = accuracy_score(y, oof_ens.argmax(1))
print(f"\n  ★ ENSEMBLE OOF Accuracy : {ens_acc:.6f}")
results["LGB+XGB+CAT Ensemble"] = ens_acc

# ─── 9. Results Summary Table ────────────────────────────────────────────────
print("\n" + "="*65)
print("  MODEL COMPARISON SUMMARY")
print("="*65)
res_df = pd.DataFrame.from_dict(results, orient="index", columns=["CV Accuracy"])
res_df = res_df.sort_values("CV Accuracy", ascending=False)
print(res_df.to_string())

# ─── 10. Confusion Matrices for ALL models ───────────────────────────────────
print("\n  Generating confusion matrices …")

X_tr, X_va, y_tr, y_va = train_test_split(
    X_sc, y, test_size=0.2, stratify=y, random_state=SEED)

baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree":       DecisionTreeClassifier(max_depth=dt_best_depth, random_state=SEED),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ax, (name, clf) in zip(axes, baseline_models.items()):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_va)
    cm = confusion_matrix(y_va, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc = accuracy_score(y_va, y_pred)
    ax.set_title(f"{name}\nVal Acc: {acc:.4f}", fontsize=11, fontweight="bold")

# Ensemble confusion matrix on OOF
ax = axes[4]
cm_ens = confusion_matrix(y, oof_ens.argmax(1))
disp_ens = ConfusionMatrixDisplay(cm_ens, display_labels=le_target.classes_)
disp_ens.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"LGB+XGB+CAT Ensemble\nOOF Acc: {ens_acc:.4f}",
             fontsize=11, fontweight="bold")

axes[5].axis("off")
plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {cm_path}")

# ─── 11. Feature Importance (LGB) ────────────────────────────────────────────
# refit lgb on full train for feature importance
lgb_full = lgb.LGBMClassifier(**lgb_params)
lgb_full.fit(X, y)
fi = pd.Series(lgb_full.feature_importances_, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(10, max(5, len(feature_cols) * 0.35)))
fi.head(30).plot.barh(color="steelblue")
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance (top 30)", fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fi_path}")

# ─── 12. Model Comparison Bar Chart ──────────────────────────────────────────
plt.figure(figsize=(10, 5))
colors = ["#e74c3c"] * (len(res_df)-1) + ["#27ae60"]
res_df["CV Accuracy"].sort_values().plot.barh(color=colors[::-1])
plt.xlabel("CV Accuracy")
plt.title("Model Comparison (K-Fold CV Accuracy)", fontweight="bold")
plt.xlim(res_df["CV Accuracy"].min() - 0.05, 1.02)
for i, v in enumerate(res_df["CV Accuracy"].sort_values()):
    plt.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
plt.tight_layout()
mc_path = os.path.join(PLOTS_DIR, "model_comparison.png")
plt.savefig(mc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {mc_path}")

# ─── 13. Generate Submission ─────────────────────────────────────────────────
print("\n" + "="*65)
print("  GENERATING SUBMISSION FILE")
print("="*65)

# Retrain ensemble on FULL training data (no folds) for final predictions
print("  Retraining LGB on full train …")
lgb_final = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": 1500})
lgb_final.fit(X, y)

print("  Retraining XGB on full train …")
final_xgb_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
xgb_final = xgb.XGBClassifier(**{**final_xgb_params, "n_estimators": 1500})
xgb_final.fit(X, y)

print("  Retraining CAT on full train …")
cat_final = cb.CatBoostClassifier(**{**cat_params, "iterations": 1500})
cat_final.fit(X, y, verbose=False)

# Final probabilities → soft vote
final_proba = (lgb_final.predict_proba(X_test)
             + xgb_final.predict_proba(X_test)
             + cat_final.predict_proba(X_test)) / 3.0

final_preds  = final_proba.argmax(axis=1)
final_labels = le_target.inverse_transform(final_preds)

sub[TARGET] = final_labels
sub.to_csv(OUT_PATH, index=False)
print(f"\n  ✔  Submission saved → '{OUT_PATH}'")
print(f"  Prediction distribution:\n{pd.Series(final_labels).value_counts().to_string()}")

# ─── 14. Classification Report ───────────────────────────────────────────────
print("\n" + "="*65)
print("  FINAL CLASSIFICATION REPORT (OOF Ensemble)")
print("="*65)
print(classification_report(
    le_target.inverse_transform(y),
    le_target.inverse_transform(oof_ens.argmax(1)),
    digits=4
))

# ─── 15. Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  DONE — FILES GENERATED")
print("="*65)
print(f"  {OUT_PATH:<40}  → submit to Kaggle")
print(f"  {cm_path:<40}  → confusion matrices")
print(f"  {fi_path:<40}  → feature importance")
print(f"  {mc_path:<40}  → model comparison")
print(f"\n  Best OOF score  : {ens_acc:.6f}")
print("="*65)