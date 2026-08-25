"""
train_headless.py — Full training pipeline from the notebook, no plots, no blocking.
Exports: attrition-api/attrition_model.joblib
         attrition-api/threshold.joblib
         attrition-api/model_meta.joblib
"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no windows, no blocking
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
candidate_paths = [
    "/mnt/data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "/mnt/data/hrfinal/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "/mnt/data/project/WA_Fn-UseC_-HR-Employee-Attrition.csv",
]
local_csv = next((p for p in candidate_paths if os.path.exists(p)), None)
if local_csv is not None:
    dataset_path = os.path.dirname(local_csv)
else:
    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    except Exception as e:
        raise FileNotFoundError(
            "Dataset not found locally and Kaggle download failed."
        ) from e

csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV file found.")

csv_path = csv_files[0]
df = pd.read_csv(csv_path)
print(f"Dataset loaded: {csv_path} — shape {df.shape}")

# ---------------------------------------------------------------------------
# 2. Data cleaning & target encoding
# ---------------------------------------------------------------------------
data = df.copy()
data = data.drop(columns=["Attrition_Binary"], errors="ignore")

constant_columns = [
    col for col in data.columns
    if col != "Attrition" and data[col].nunique(dropna=False) <= 1
]
id_like_columns = [
    col for col in data.columns
    if col != "Attrition" and data[col].nunique(dropna=False) == len(data)
]
auto_drop_columns = sorted(set(constant_columns + id_like_columns))
data = data.drop(columns=auto_drop_columns, errors="ignore")
data["Attrition"] = data["Attrition"].map({"No": 0, "Yes": 1})
print(f"Dropped constant/ID-like columns: {auto_drop_columns}")
print(f"Final raw modeling shape: {data.shape}")

# ---------------------------------------------------------------------------
# 3. Feature engineering  (must match feature_engineering.py exactly)
# ---------------------------------------------------------------------------
def add_hr_features(X):
    X = X.copy()
    X["YearsSincePromotionRatio"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
    X["CurrentRoleRatio"]         = X["YearsInCurrentRole"]       / (X["YearsAtCompany"] + 1)
    X["ManagerTenureRatio"]       = X["YearsWithCurrManager"]     / (X["YearsAtCompany"] + 1)
    X["CompanyExperienceRatio"]   = X["YearsAtCompany"]           / (X["TotalWorkingYears"] + 1)
    X["IncomePerJobLevel"]        = X["MonthlyIncome"]            / (X["JobLevel"] + 1)
    X["IncomePerYearExperience"]  = X["MonthlyIncome"]            / (X["TotalWorkingYears"] + 1)
    X["YearsAtCompanyRatioToAge"] = X["YearsAtCompany"]           / (X["Age"] + 1)
    X["JobLevelPerYear"]          = X["JobLevel"]                 / (X["TotalWorkingYears"] + 1)
    satisfaction_cols = [
        "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance"
    ]
    X["SatisfactionAvg"] = X[satisfaction_cols].mean(axis=1)
    X["WorkPressureScore"] = (
        X["OverTime"].map({"Yes": 1, "No": 0})
        + (4 - X["WorkLifeBalance"]) / 4
        + (4 - X["JobSatisfaction"]) / 4
    )
    X["EarlyCareer"]        = ((X["Age"] < 30) & (X["TotalWorkingYears"] < 5)).astype(int)
    X["FrequentJobChanges"] = ((X["NumCompaniesWorked"] >= 3) & (X["TotalWorkingYears"] <= 10)).astype(int)
    X["OverTime_JobSatisfaction"] = X["OverTime"].astype(str) + "_" + X["JobSatisfaction"].astype(str)
    X["OverTime_WorkLifeBalance"] = X["OverTime"].astype(str) + "_" + X["WorkLifeBalance"].astype(str)
    X["OverTime_JobLevel"]        = X["OverTime"].astype(str) + "_" + X["JobLevel"].astype(str)
    X["Marital_OverTime"]         = X["MaritalStatus"].astype(str) + "_" + X["OverTime"].astype(str)
    X["JobRole_OverTime"]         = X["JobRole"].astype(str) + "_" + X["OverTime"].astype(str)
    return X

X = add_hr_features(data.drop(columns=["Attrition"]))
y = data["Attrition"]
print(f"Features after engineering: {X.shape[1]}")

# ---------------------------------------------------------------------------
# 4. 70/15/15 stratified split
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=(0.15 / 0.85), random_state=42, stratify=y_train_val
)
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 5. Preprocessors
# ---------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_features     = X_train.select_dtypes(exclude=["object"]).columns.tolist()

linear_preprocessor_final = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_features)
])

tree_preprocessor_final = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_features)
])

cat_indices = [X_train.columns.get_loc(c) for c in categorical_features]
print(f"Categorical features: {len(categorical_features)} | Numeric: {len(numeric_features)}")

# ---------------------------------------------------------------------------
# 6. Imports for models and imbalance handling
# ---------------------------------------------------------------------------
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import (
    SMOTE, SMOTENC, BorderlineSMOTE, ADASYN, RandomOverSampler
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 7. Scoring & threshold selection helpers
# ---------------------------------------------------------------------------
def score_probabilities(y_true, prob, threshold=0.50):
    y_true = np.asarray(y_true)
    prob   = np.asarray(prob)
    pred   = (prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "Accuracy":          accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision":         precision_score(y_true, pred, zero_division=0),
        "Recall":            recall_score(y_true, pred, zero_division=0),
        "F1":                f1_score(y_true, pred, zero_division=0),
        "ROC-AUC":           roc_auc_score(y_true, prob),
        "PR-AUC":            average_precision_score(y_true, prob),
        "Predicted Yes %":   pred.mean(),
        "Actual Yes %":      y_true.mean(),
        "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]
    }

def choose_threshold(y_true, prob):
    rows = []
    for t in np.round(np.arange(0.10, 0.901, 0.01), 2):
        rows.append({"Threshold": t, **score_probabilities(y_true, prob, t)})
    d = pd.DataFrame(rows).sort_values(
        ["F1", "Balanced Accuracy", "PR-AUC", "Recall", "Precision"], ascending=False
    )
    return float(d.iloc[0]["Threshold"]), d

# ---------------------------------------------------------------------------
# 8. Model builders
# ---------------------------------------------------------------------------
def base_classifier(model_name, variant="baseline"):
    if model_name == "Logistic Regression":
        cw = (None if variant == "baseline"
              else ({0: 1, 1: float(variant.split("=")[1])} if variant.startswith("weight=")
                    else "balanced"))
        clf = LogisticRegression(C=0.5, max_iter=5000, class_weight=cw, random_state=RANDOM_STATE)
        return Pipeline([("preprocessor", linear_preprocessor_final), ("classifier", clf)])

    if model_name == "Decision Tree":
        cw = None if variant == "baseline" else "balanced"
        clf = DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=4,
                                     class_weight=cw, random_state=RANDOM_STATE)
        return Pipeline([("preprocessor", tree_preprocessor_final), ("classifier", clf)])

    if model_name == "Random Forest":
        cw = None if variant == "baseline" else variant
        clf = RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_split=5,
                                     min_samples_leaf=2, max_features="sqrt",
                                     class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)
        return Pipeline([("preprocessor", tree_preprocessor_final), ("classifier", clf)])

    if model_name == "Extra Trees":
        cw = None if variant == "baseline" else "balanced"
        clf = ExtraTreesClassifier(n_estimators=250, max_depth=12, min_samples_leaf=2,
                                   max_features="sqrt", class_weight=cw,
                                   random_state=RANDOM_STATE, n_jobs=-1)
        return Pipeline([("preprocessor", tree_preprocessor_final), ("classifier", clf)])

    if model_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, max_depth=2,
                                         min_samples_split=10, min_samples_leaf=4,
                                         random_state=RANDOM_STATE)
        return Pipeline([("preprocessor", tree_preprocessor_final), ("classifier", clf)])

    if model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=250,
                                             l2_regularization=1.0, random_state=RANDOM_STATE)
        return Pipeline([("preprocessor", tree_preprocessor_final), ("classifier", clf)])

    if model_name == "XGBoost":
        spw = 1.0 if variant == "baseline" else float(variant.split("=")[1])
        clf = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03,
                            min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
                            reg_lambda=2, scale_pos_weight=spw,
                            objective="binary:logistic", eval_metric="logloss",
                            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
                            verbosity=0)
        return Pipeline([("preprocessor", tree_preprocessor_final), ("classifier", clf)])

    if model_name == "SVM":
        cw = None if variant == "baseline" else "balanced"
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                  class_weight=cw, random_state=RANDOM_STATE)
        return Pipeline([("preprocessor", linear_preprocessor_final), ("classifier", clf)])

    if model_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=15, weights="distance", p=2)
        return Pipeline([("preprocessor", linear_preprocessor_final), ("classifier", clf)])

    raise ValueError(model_name)


def resampled_pipeline(model_name, kind, ratio):
    pre = (linear_preprocessor_final
           if model_name in ["Logistic Regression", "SVM", "KNN"]
           else tree_preprocessor_final)
    clf = base_classifier(model_name).named_steps["classifier"]

    if kind == "SMOTE":
        sampler = SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE, k_neighbors=5)
    elif kind == "SMOTENC":
        sampler = SMOTENC(categorical_features=cat_indices, sampling_strategy=ratio,
                          random_state=RANDOM_STATE, k_neighbors=5)
        return ImbPipeline([("sampler", sampler), ("preprocessor", pre), ("classifier", clf)])
    elif kind == "BorderlineSMOTE":
        sampler = BorderlineSMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE, k_neighbors=5)
    elif kind == "ADASYN":
        sampler = ADASYN(sampling_strategy=ratio, random_state=RANDOM_STATE, n_neighbors=5)
    elif kind == "RandomOverSampler":
        sampler = RandomOverSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    elif kind == "RandomUnderSampler":
        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    elif kind == "SMOTETomek":
        sampler = SMOTETomek(sampling_strategy=ratio, random_state=RANDOM_STATE)
    elif kind == "SMOTEENN":
        sampler = SMOTEENN(sampling_strategy=ratio, random_state=RANDOM_STATE)
    else:
        raise ValueError(kind)

    return ImbPipeline([("preprocessor", pre), ("sampler", sampler), ("classifier", clf)])


# ---------------------------------------------------------------------------
# 9. Build all configs
# ---------------------------------------------------------------------------
base_models = [
    "Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees",
    "Gradient Boosting", "HistGradientBoosting", "XGBoost", "SVM", "KNN"
]

configs = []
for m in base_models:
    configs.append((m, "Baseline", base_classifier(m)))

for w in [1.5, 2.0, 2.5, 3.0]:
    configs.append(("Logistic Regression", f"ClassWeight {w:g}",
                     base_classifier("Logistic Regression", f"weight={w}")))
configs += [
    ("Decision Tree",  "ClassWeight balanced",            base_classifier("Decision Tree",  "balanced")),
    ("Random Forest",  "ClassWeight balanced",            base_classifier("Random Forest",  "balanced")),
    ("Random Forest",  "ClassWeight balanced_subsample",  base_classifier("Random Forest",  "balanced_subsample")),
    ("Extra Trees",    "ClassWeight balanced",            base_classifier("Extra Trees",    "balanced")),
    ("SVM",            "ClassWeight balanced",            base_classifier("SVM",            "balanced")),
]
for spw in [1.5, 2.0, 2.5, 3.0]:
    configs.append(("XGBoost", f"scale_pos_weight {spw:g}", base_classifier("XGBoost", f"spw={spw}")))

for m in base_models:
    for kind in (["SMOTE", "SMOTENC"] if m not in ["Gradient Boosting", "HistGradientBoosting", "KNN"] else ["SMOTE"]):
        configs.append((m, f"{kind} 0.60", resampled_pipeline(m, kind, 0.60)))
    if m in ["Gradient Boosting", "HistGradientBoosting", "KNN"]:
        configs.append((m, "SMOTE 0.80", resampled_pipeline(m, "SMOTE", 0.80)))

for kind in ["BorderlineSMOTE", "ADASYN", "SMOTETomek", "SMOTEENN", "RandomOverSampler"]:
    configs.append(("Logistic Regression", f"{kind} 0.60",
                     resampled_pipeline("Logistic Regression", kind, 0.60)))

print(f"Total candidates: {len(configs)}")

# ---------------------------------------------------------------------------
# 10. OOF screening
# ---------------------------------------------------------------------------
SCREEN_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def oof_predictions(pipeline, X_data, y_data, cv=SCREEN_CV):
    X_data = X_data.reset_index(drop=True)
    y_arr  = np.asarray(y_data)
    prob_sum   = np.zeros(len(y_arr))
    prob_count = np.zeros(len(y_arr))
    fold_rows  = []
    for fold, (tr, va) in enumerate(cv.split(X_data, y_arr), 1):
        model = clone(pipeline)
        model.fit(X_data.iloc[tr], y_arr[tr])
        prob = model.predict_proba(X_data.iloc[va])[:, 1]
        prob_sum[va]   += prob
        prob_count[va] += 1
        fold_rows.append(score_probabilities(y_arr[va], prob, 0.50))
    oof = prob_sum / np.maximum(prob_count, 1)
    return oof, pd.DataFrame(fold_rows)

results   = []
oof_store = {}
for i, (model_name, strategy, pipeline) in enumerate(configs, 1):
    try:
        oof, fold_df  = oof_predictions(pipeline, X_train, y_train)
        threshold, _  = choose_threshold(y_train.values, oof)
        metrics       = score_probabilities(y_train.values, oof, threshold)
        results.append({"Model": model_name, "Strategy": strategy,
                         "Threshold": threshold, **metrics,
                         "CV F1 SD": fold_df["F1"].std(),
                         "CV Balanced Accuracy SD": fold_df["Balanced Accuracy"].std()})
        oof_store[(model_name, strategy)] = oof
        print(f"[{i:02d}/{len(configs)}] {model_name:22s} | {strategy:30s} "
              f"| F1={metrics['F1']:.3f} | Recall={metrics['Recall']:.3f} "
              f"| PR-AUC={metrics['PR-AUC']:.3f}")
    except Exception as e:
        print(f"[{i:02d}/{len(configs)}] FAILED {model_name} | {strategy}: {e}")

cv_results = pd.DataFrame(results).sort_values(
    ["F1", "PR-AUC", "Balanced Accuracy", "Recall", "Precision"], ascending=False
).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 11. Repeated-CV refinement (top 2 per family)
# ---------------------------------------------------------------------------
def repeated_oof(pipeline, X_data, y_data):
    cv     = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    X_data = X_data.reset_index(drop=True)
    y_arr  = np.asarray(y_data)
    ps = np.zeros(len(y_arr)); pc = np.zeros(len(y_arr)); fold_rows = []
    for fold, (tr, va) in enumerate(cv.split(X_data, y_arr), 1):
        model = clone(pipeline)
        model.fit(X_data.iloc[tr], y_arr[tr])
        prob = model.predict_proba(X_data.iloc[va])[:, 1]
        ps[va] += prob; pc[va] += 1
        fold_rows.append(score_probabilities(y_arr[va], prob, 0.50))
    return ps / pc, pd.DataFrame(fold_rows)

screen_top = cv_results.groupby("Model", group_keys=False).head(2)
refined    = []
for _, r in screen_top.iterrows():
    pipe = next(p for m, s, p in configs if m == r["Model"] and s == r["Strategy"])
    oof, fold_df  = repeated_oof(pipe, X_train, y_train)
    threshold, _  = choose_threshold(y_train.values, oof)
    metrics       = score_probabilities(y_train.values, oof, threshold)
    refined.append({"Model": r["Model"], "Strategy": r["Strategy"],
                     "Threshold": threshold, **metrics,
                     "CV F1 SD": fold_df["F1"].std(),
                     "CV Balanced Accuracy SD": fold_df["Balanced Accuracy"].std()})
    oof_store[(r["Model"], r["Strategy"])] = oof
    print(f"  [Refined] {r['Model']:22s} | {r['Strategy']:30s} | F1={metrics['F1']:.3f}")

refined_df  = pd.DataFrame(refined)
keys        = set(zip(refined_df["Model"], refined_df["Strategy"]))
base_unref  = cv_results[~cv_results.apply(
    lambda row: (row["Model"], row["Strategy"]) in keys, axis=1)]
cv_results  = pd.concat([base_unref, refined_df], ignore_index=True).sort_values(
    ["F1", "PR-AUC", "Balanced Accuracy", "Recall", "Precision"], ascending=False
).reset_index(drop=True)

best_per_model = (
    cv_results.sort_values(["F1", "PR-AUC", "Balanced Accuracy", "Recall", "Precision"],
                            ascending=False)
    .groupby("Model", as_index=False).first()
    .sort_values(["F1", "PR-AUC"], ascending=False)
)
print("\nBest per model family:")
print(best_per_model[["Model", "Strategy", "Threshold", "F1", "Recall", "PR-AUC",
                        "Balanced Accuracy"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 12. Tuned XGBoost (Model 10) — nested CV
# ---------------------------------------------------------------------------
print("\n--- Tuned XGBoost nested CV ---")
xgb_tune = Pipeline([
    ("preprocessor", tree_preprocessor_final),
    ("classifier",   XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", verbosity=0
    ))
])
xgb_params = {
    "classifier__n_estimators":      [250, 350, 450, 550],
    "classifier__max_depth":         [2, 3, 4],
    "classifier__learning_rate":     [0.02, 0.03, 0.05, 0.07],
    "classifier__min_child_weight":  [1, 2, 4, 6],
    "classifier__subsample":         [0.8, 0.9, 1.0],
    "classifier__colsample_bytree":  [0.8, 0.9, 1.0],
    "classifier__reg_alpha":         [0, 0.05, 0.2],
    "classifier__reg_lambda":        [1, 2, 4, 6],
    "classifier__scale_pos_weight":  [1.5, 2.0, 2.5, 3.0],
}
outer      = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
tuned_oof  = np.zeros(len(X_train))
for fold, (tr, va) in enumerate(outer.split(X_train, y_train), 1):
    inner_search = RandomizedSearchCV(
        clone(xgb_tune), xgb_params, n_iter=12, scoring="f1",
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE + fold),
        random_state=RANDOM_STATE + fold, n_jobs=-1, refit=True
    )
    inner_search.fit(X_train.iloc[tr], y_train.iloc[tr])
    tuned_oof[va] = inner_search.best_estimator_.predict_proba(X_train.iloc[va])[:, 1]
    print(f"  Nested XGBoost fold {fold}/5 done.")

tuned_threshold, _ = choose_threshold(y_train.values, tuned_oof)
tuned_metrics      = score_probabilities(y_train.values, tuned_oof, tuned_threshold)
oof_store[("Tuned XGBoost", "tuned")] = tuned_oof
print(f"  Tuned XGBoost — F1={tuned_metrics['F1']:.4f} | Recall={tuned_metrics['Recall']:.4f} | Threshold={tuned_threshold}")

# Final full-train search for eventual refit
search = RandomizedSearchCV(
    xgb_tune, xgb_params, n_iter=20, scoring="f1",
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE, n_jobs=-1, refit=True
)
search.fit(X_train, y_train)
tuned_xgb = search.best_estimator_
print(f"  Best XGB params: {search.best_params_}")

# ---------------------------------------------------------------------------
# 13. Probability Ensemble (Model 11)
# ---------------------------------------------------------------------------
print("\n--- Building probability ensemble ---")
rows_by_model = best_per_model.set_index("Model")
families = [m for m in ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost", "SVM"]
            if m in rows_by_model.index][:3]

components = []
for fam in families:
    strat = rows_by_model.loc[fam, "Strategy"]
    key   = (fam, strat)
    if key in oof_store:
        components.append((fam, strat, oof_store[key]))
components.append(("Tuned XGBoost", "tuned", oof_store[("Tuned XGBoost", "tuned")]))

probs = [c[2] for c in components]
names = [c[0] for c in components]
print(f"  Ensemble components: {names}")

weight_grid = []
if len(probs) == 2:
    for w in np.arange(0, 1.01, 0.05):
        weight_grid.append([w, 1 - w])
elif len(probs) == 3:
    for w1 in np.arange(0, 1.01, 0.10):
        for w2 in np.arange(0, 1.01 - w1, 0.10):
            weight_grid.append([w1, w2, 1 - w1 - w2])
elif len(probs) == 4:
    for w1 in np.arange(0, 0.81, 0.20):
        for w2 in np.arange(0, 0.81 - w1, 0.20):
            for w3 in np.arange(0, 0.81 - w1 - w2, 0.20):
                w4 = 1 - w1 - w2 - w3
                if w4 >= 0:
                    weight_grid.append([w1, w2, w3, w4])

blend_rows = []
for weights in weight_grid:
    blend = sum(w * p for w, p in zip(weights, probs))
    t, _  = choose_threshold(y_train.values, blend)
    blend_rows.append({"Weights": weights, "Threshold": t,
                        **score_probabilities(y_train.values, blend, t)})

blend_df         = pd.DataFrame(blend_rows).sort_values(
    ["F1", "PR-AUC", "Balanced Accuracy", "Recall", "Precision"], ascending=False)
best_blend       = blend_df.iloc[0]
ensemble_oof     = sum(w * p for w, p in zip(best_blend["Weights"], probs))
ensemble_threshold = float(best_blend["Threshold"])
ensemble_metrics = score_probabilities(y_train.values, ensemble_oof, ensemble_threshold)
print(f"  Ensemble — F1={ensemble_metrics['F1']:.4f} | Threshold={ensemble_threshold}")

# ---------------------------------------------------------------------------
# 14. Final leaderboard & rigorous outer-CV selection
# ---------------------------------------------------------------------------
print("\n--- Rigorous outer-CV model selection ---")

def nested_threshold_eval(pipeline, X_data, y_data, outer_splits=5, inner_splits=4):
    X_data = X_data.reset_index(drop=True); y_arr = np.asarray(y_data)
    outer  = StratifiedKFold(outer_splits, shuffle=True, random_state=RANDOM_STATE + 101)
    rows   = []
    for fold, (tr, va) in enumerate(outer.split(X_data, y_arr), 1):
        inner      = StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE + 1000 + fold)
        inner_prob = np.zeros(len(tr)); inner_count = np.zeros(len(tr))
        for itr, iva in inner.split(X_data.iloc[tr], y_arr[tr]):
            m = clone(pipeline); m.fit(X_data.iloc[tr].iloc[itr], y_arr[tr][itr])
            inner_prob[iva] = m.predict_proba(X_data.iloc[tr].iloc[iva])[:, 1]
            inner_count[iva] += 1
        t, _ = choose_threshold(y_arr[tr], inner_prob / np.maximum(inner_count, 1))
        m    = clone(pipeline); m.fit(X_data.iloc[tr], y_arr[tr])
        p    = m.predict_proba(X_data.iloc[va])[:, 1]
        met  = score_probabilities(y_arr[va], p, t)
        met["Fold"] = fold; met["Threshold"] = t; rows.append(met)
    d = pd.DataFrame(rows)
    return {
        "F1": float(d["F1"].mean()), "Recall": float(d["Recall"].mean()),
        "Precision": float(d["Precision"].mean()), "PR-AUC": float(d["PR-AUC"].mean()),
        "Balanced Accuracy": float(d["Balanced Accuracy"].mean()),
        "ROC-AUC": float(d["ROC-AUC"].mean()), "Accuracy": float(d["Accuracy"].mean()),
        "F1 SD": float(d["F1"].std()), "Threshold mean": float(d["Threshold"].mean()),
    }, d

rigorous_rows = []
for _, r in best_per_model.iterrows():
    pipe = next(p for m, s, p in configs if m == r["Model"] and s == r["Strategy"])
    met, _ = nested_threshold_eval(pipe, X_train, y_train)
    rigorous_rows.append({"Model": r["Model"], "Strategy": r["Strategy"], **met})
    print(f"  Rigorous CV: {r['Model']:22s} | F1={met['F1']:.4f} | Recall={met['Recall']:.4f}")

rigorous_df = pd.DataFrame(rigorous_rows).sort_values(
    ["F1", "PR-AUC", "Balanced Accuracy", "Recall", "Precision"], ascending=False
).reset_index(drop=True)

# Freeze final selection
rigorous_best  = rigorous_df.iloc[0]
final_model_name = str(rigorous_best["Model"])
final_strategy   = str(rigorous_best["Strategy"])
final_pipeline   = next(p for m, s, p in configs if m == final_model_name and s == final_strategy)

# Freeze threshold on dev set (train + val) with inner CV
X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_dev = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
inner_cv   = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 909)
inner_prob = np.zeros(len(X_dev)); inner_count = np.zeros(len(X_dev))
for tr, va in inner_cv.split(X_dev, y_dev):
    m = clone(final_pipeline); m.fit(X_dev.iloc[tr], y_dev.iloc[tr])
    inner_prob[va] = m.predict_proba(X_dev.iloc[va])[:, 1]; inner_count[va] += 1
final_threshold, _ = choose_threshold(y_dev.values, inner_prob / np.maximum(inner_count, 1))

print(f"\n{'='*70}")
print(f"FINAL SELECTED MODEL : {final_model_name}")
print(f"STRATEGY             : {final_strategy}")
print(f"FROZEN THRESHOLD     : {final_threshold}")
print(f"{'='*70}")

# ---------------------------------------------------------------------------
# 15. Final refit on dev (train + val) and test evaluation
# ---------------------------------------------------------------------------
final_fitted_model = clone(final_pipeline)
final_fitted_model.fit(X_dev, y_dev)
test_prob   = final_fitted_model.predict_proba(X_test)[:, 1]
test_pred   = (test_prob >= final_threshold).astype(int)
final_test_metrics = score_probabilities(y_test.values, test_prob, final_threshold)

print("\nFINAL UNTOUCHED TEST RESULT")
print("=" * 70)
for k in ["Accuracy", "Balanced Accuracy", "Precision", "Recall",
          "F1", "ROC-AUC", "PR-AUC", "Predicted Yes %", "Actual Yes %"]:
    print(f"  {k:22s}: {final_test_metrics[k]:.4f}")

print("\n" + classification_report(
    y_test, test_pred, target_names=["Stayed/No", "Left/Yes"], zero_division=0
))
print(f"TN={final_test_metrics['TN']} | FP={final_test_metrics['FP']} "
      f"| FN={final_test_metrics['FN']} | TP={final_test_metrics['TP']}")

# ---------------------------------------------------------------------------
# 16. Export
# ---------------------------------------------------------------------------
import joblib
os.makedirs("attrition-api", exist_ok=True)

joblib.dump(final_fitted_model, "attrition-api/attrition_model.joblib")
joblib.dump(final_threshold,    "attrition-api/threshold.joblib")

model_meta = {
    "model_name":            final_model_name,
    "model_strategy":        final_strategy,
    "threshold":             final_threshold,
    "test_f1":               round(final_test_metrics["F1"],               4),
    "test_recall":           round(final_test_metrics["Recall"],           4),
    "test_precision":        round(final_test_metrics["Precision"],        4),
    "test_pr_auc":           round(final_test_metrics["PR-AUC"],           4),
    "test_roc_auc":          round(final_test_metrics["ROC-AUC"],          4),
    "test_balanced_accuracy":round(final_test_metrics["Balanced Accuracy"],4),
    "test_accuracy":         round(final_test_metrics["Accuracy"],         4),
}
joblib.dump(model_meta, "attrition-api/model_meta.joblib")

print("\n" + "=" * 70)
print("EXPORTED:")
print(f"  attrition-api/attrition_model.joblib")
print(f"  attrition-api/threshold.joblib")
print(f"  attrition-api/model_meta.joblib")
print(f"  Selected model : {final_model_name}")
print(f"  Strategy       : {final_strategy}")
print(f"  Threshold      : {final_threshold:.4f}")
print(f"  Test F1        : {final_test_metrics['F1']:.4f}")
print(f"  Test Recall    : {final_test_metrics['Recall']:.4f}")
print(f"  Test PR-AUC    : {final_test_metrics['PR-AUC']:.4f}")
print("=" * 70)
