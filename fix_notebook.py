import json

NOTEBOOK = "Group09_HR_attrition_FINAL_Submission_File.ipynb"

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

# The guard must check for ALL variables Cell 57 depends on.
# If any upstream variable is missing, raise a clear, actionable error
# instead of a confusing NameError deep inside the cell.
NEW_SOURCE = """\
X_dev=pd.concat([X_train,X_val],axis=0).reset_index(drop=True)
y_dev=pd.concat([y_train,y_val],axis=0).reset_index(drop=True)

# ── Dependency guard ────────────────────────────────────────────────────────
# Check every variable this cell needs and raise a clear message if any is missing.
_missing = [name for name in [
    "final_model_name", "final_strategy", "final_threshold",
    "base_models", "configs", "tuned_xgb", "best_blend", "components"
] if name not in dir()]

if _missing:
    # Try to recover final_model_name / final_strategy / final_threshold from rigorous_df
    _recoverable = {"final_model_name", "final_strategy", "final_threshold"}
    _still_missing = [v for v in _missing if v not in _recoverable]

    if _still_missing:
        raise RuntimeError(
            f"The following required variables are not defined: {_still_missing}\\n"
            "These come from earlier cells that must be run first.\\n"
            "Please use Kernel → Restart & Run All, or manually run cells in order:\\n"
            "  base_models  → Cell 39\\n"
            "  tuned_xgb    → Cell 44\\n"
            "  best_blend / components → Cell 46\\n"
            "  rigorous_df  → Cell 54\\n"
            "  final_model_name / final_threshold → Cell 52"
        )

    # Only the final selection vars are missing — recover them from rigorous_df
    print("[Guard] Recovering final_model_name / final_strategy / final_threshold from rigorous_df ...")
    rigorous_best = rigorous_df.iloc[0]
    final_model_name = str(rigorous_best["Model"])
    final_strategy   = str(rigorous_best["Strategy"])
    _pipe = next(p for m,s,p in configs if m==final_model_name and s==final_strategy)
    _inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE+909)
    _inner_prob = np.zeros(len(X_dev)); _inner_count = np.zeros(len(X_dev))
    for _tr,_va in _inner.split(X_dev, y_dev):
        _m = clone(_pipe); _m.fit(X_dev.iloc[_tr], y_dev.iloc[_tr])
        _inner_prob[_va] += _m.predict_proba(X_dev.iloc[_va])[:,1]
        _inner_count[_va] += 1
    final_threshold, _ = choose_threshold(y_dev.values, _inner_prob/np.maximum(_inner_count,1))
    print(f"[Guard] Restored → model='{final_model_name}' | strategy='{final_strategy}' | threshold={final_threshold:.2f}")
# ────────────────────────────────────────────────────────────────────────────

if final_model_name in base_models:
    final_pipeline=next(p for m,s,p in configs if m==final_model_name and s==final_strategy)
    final_fitted_model=clone(final_pipeline)
    final_fitted_model.fit(X_dev,y_dev)
    test_prob=final_fitted_model.predict_proba(X_test)[:,1]

elif final_model_name=="Tuned XGBoost":
    final_fitted_model=clone(tuned_xgb)
    final_fitted_model.fit(X_dev,y_dev)
    test_prob=final_fitted_model.predict_proba(X_test)[:,1]

else:
    fitted=[]
    for fam,strat,_ in components:
        if fam=="Tuned XGBoost":
            p=clone(tuned_xgb)
        else:
            p=next(p for m,s,p in configs if m==fam and s==strat)
        p.fit(X_dev,y_dev)
        fitted.append(p)
    component_probs=[p.predict_proba(X_test)[:,1] for p in fitted]
    test_prob=sum(w*p for w,p in zip(best_blend["Weights"],component_probs))

test_pred=(test_prob>=final_threshold).astype(int)
final_test_metrics=score_probabilities(y_test.values,test_prob,final_threshold)

print("="*70)
print("FINAL UNTOUCHED TEST RESULT")
print("="*70)
for k in ["Accuracy","Balanced Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC","Predicted Yes %","Actual Yes %"]:
    print(f"{k:22s}: {final_test_metrics[k]:.4f}")

print("\\nClassification report:")
print(classification_report(y_test,test_pred,target_names=["Stayed/No","Left/Yes"],zero_division=0))
print(f"TN={final_test_metrics['TN']} | FP={final_test_metrics['FP']} | FN={final_test_metrics['FN']} | TP={final_test_metrics['TP']}")
print(f"Actual Left cases: {final_test_metrics['TP']+final_test_metrics['FN']}")
print(f"Left cases detected: {final_test_metrics['TP']} ({final_test_metrics['Recall']:.1%})")
"""

cell = nb["cells"][57]
assert cell["cell_type"] == "code"
cell["source"] = NEW_SOURCE.splitlines(keepends=True)
print("Cell 57 patched.")

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
