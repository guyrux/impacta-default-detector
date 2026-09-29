# %%
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

FIG = "./figures"
ART = "./artifacts"
# FIG.mkdir(exist_ok=True)
# ART.mkdir(exist_ok=True)
CSV = "./data/raw/credit_risk_dataset.csv"

df = pd.read_csv(CSV, sep=";", encoding="utf-8-sig")
orig_cols = list(df.columns)
rename = {
    "idade": "age",
    "renda": "income",
    "tipo de propriedade da residência": "home_ownership",
    "tempo de emprego": "employment_length",
    "finalidade do empréstimo": "loan_intent",
    "classificação do empréstimo": "loan_grade",
    "valor do empréstimo": "loan_amount",
    "taxa de juros do empréstimo": "interest_rate",
    "status do empréstimo": "loan_status",
    "Percentual da renda comprometido com o empréstimo": "loan_percent_income",
    "histórico de inadimplência no cadastro de crédito": "cb_default_on_file",
    "Tempo de histórico de crédito": "credit_history_length",
}
df = df.rename(columns=rename)
num = [
    "age",
    "income",
    "employment_length",
    "loan_amount",
    "interest_rate",
    "loan_percent_income",
    "credit_history_length",
]
cat = ["home_ownership", "loan_intent", "loan_grade", "cb_default_on_file"]
target = "loan_status"

# qualidade
missing = df.isna().sum().to_dict()
duplicates = int(df.duplicated().sum())
invalid = {
    "age_nonpositive": int((df.age <= 0).sum()),
    "income_nonpositive": int((df.income <= 0).sum()),
    "employment_negative": int((df.employment_length < 0).sum()),
    "loan_amount_nonpositive": int((df.loan_amount <= 0).sum()),
    "interest_rate_nonpositive": int((df.interest_rate <= 0).sum()),
    "loan_percent_income_outside_0_1": int(
        ((df.loan_percent_income < 0) | (df.loan_percent_income > 1)).sum()
    ),
    "credit_history_negative": int((df.credit_history_length < 0).sum()),
    "target_not_binary": int((~df[target].isin([0, 1])).sum()),
}
summary = df[num].describe().round(4).to_dict()
cardinality = {
    c: sorted([str(x) for x in df[c].dropna().unique().tolist()])
    for c in cat + [target]
}
class_dist = df[target].value_counts().sort_index().to_dict()

# outliers IQR
outliers = {}
for c in num:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (df[c] < lo) | (df[c] > hi)
    outliers[c] = {
        "count": int(mask.sum()),
        "pct": round(float(mask.mean() * 100), 2),
        "lower": float(lo),
        "upper": float(hi),
    }

# relacionamento alvo
rates_by_cat = {}
for c in cat:
    tab = df.groupby(c)[target].agg(["count", "mean"]).round(4)
    rates_by_cat[c] = tab.reset_index().to_dict(orient="records")
# correlaçoes númericas com o alvo
corr = (
    df[num + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
    .round(4)
    .to_dict()
)

# Tipos de gráficos
sns.set_theme(style="whitegrid", context="notebook")
plt.figure(figsize=(7, 4.5))
sns.countplot(data=df, x=target, color="#2563eb")
plt.title("Distribuição do status do empréstimo")
plt.xlabel("Status (0 = classe 0; 1 = evento positivo)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(f"{FIG}/01_target_distribution.png", dpi=180)
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, c in zip(
    axes.flat, ["income", "loan_amount", "interest_rate", "loan_percent_income"]
):
    sns.boxplot(data=df, x=target, y=c, ax=ax, color="#60a5fa")
    ax.set_title(c)
    ax.set_xlabel("Status do empréstimo")
plt.tight_layout()
plt.savefig(f"{FIG}/02_numeric_by_target.png", dpi=180)
plt.close()
plt.figure(figsize=(9, 6))
sns.heatmap(df[num + [target]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig(f"{FIG}/03_correlation.png", dpi=180)
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, c in zip(
    axes.flat, ["loan_grade", "home_ownership", "loan_intent", "cb_default_on_file"]
):
    rates = df.groupby(c)[target].mean().sort_values(ascending=False)
    sns.barplot(x=rates.values, y=rates.index, ax=ax, color="#0f766e")
    ax.set_title("Taxa de pagamento por " + c)
    ax.set_xlabel("Média de loan_status")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{FIG}/04_categorical_payment_rates.png", dpi=180)
plt.close()

# modelagem
X = df.drop(columns=[target])
y = df[target].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pre = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
            ),
            num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat,
        ),
    ]
)
models = {
    "Regressão Logística": LogisticRegression(
        max_iter=1500, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}
results = {}
fitted = {}
curves = []
for name, est in models.items():
    pipe = Pipeline([("prep", pre), ("model", est)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "average_precision": round(float(average_precision_score(y_test, prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    fitted[name] = pipe
    fpr, tpr, _ = roc_curve(y_test, prob)
    curves.append((name, fpr, tpr, results[name]["roc_auc"]))
plt.figure(figsize=(7, 5))
for name, fpr, tpr, auc in curves:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Taxa de falsos positivos")
plt.ylabel("Taxa de verdadeiros positivos")
plt.title("Curvas ROC no conjunto de teste")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/05_roc_curves.png", dpi=180)
plt.close()
# treinamento para estabilidade
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_summary = {}
for name, est in models.items():
    pipe = Pipeline([("prep", pre), ("model", est)])
    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "recall", "precision"],
        n_jobs=-1,
    )
    cv_summary[name] = {
        m: {
            "mean": round(float(scores["test_" + m].mean()), 4),
            "std": round(float(scores["test_" + m].std()), 4),
        }
        for m in ["roc_auc", "f1", "recall", "precision"]
    }
# importância de permutação com base no melhor ROC AUC
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best = fitted[best_name]
perm = permutation_importance(
    best, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
)
importance = sorted(
    [
        {
            "feature": c,
            "mean_decrease_auc": round(float(m), 5),
            "std": round(float(s), 5),
        }
        for c, m, s in zip(X.columns, perm.importances_mean, perm.importances_std)
    ],
    key=lambda z: z["mean_decrease_auc"],
    reverse=True,
)
plt.figure(figsize=(8, 5))
top = importance[:10][::-1]
plt.barh(
    [z["feature"] for z in top], [z["mean_decrease_auc"] for z in top], color="#7c3aed"
)
plt.xlabel("Queda média no ROC-AUC")
plt.title("Importância por permutação — " + best_name)
plt.tight_layout()
plt.savefig(f"{FIG}/06_permutation_importance.png", dpi=180)
plt.close()
# diagnósticos de subgrupo no limiar padrão
subgroup = {}
for c in ["cb_default_on_file", "loan_grade", "home_ownership"]:
    rows = []
    for level, g in df.loc[X_test.index].assign(_y=y_test).groupby(c):
        idx = g.index
        pred = fitted[best_name].predict(X.loc[idx])
        actual = y.loc[idx]
        rows.append(
            {
                "group": str(level),
                "n": int(len(idx)),
                "positive_rate_actual": round(float(actual.mean()), 4),
                "positive_rate_predicted": round(float(pred.mean()), 4),
                "recall": round(float(recall_score(actual, pred, zero_division=0)), 4),
            }
        )
    subgroup[c] = rows

output = {
    "dataset": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "original_columns": orig_cols,
        "normalized_columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    },
    "quality": {"missing": missing, "duplicates": duplicates, "invalid": invalid},
    "descriptive_statistics": summary,
    "cardinality": cardinality,
    "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
    "outliers_iqr": outliers,
    "target_rates_by_category": rates_by_cat,
    "numeric_correlations_with_target": corr,
    "models_test": results,
    "models_cv": cv_summary,
    "best_model": best_name,
    "permutation_importance": importance,
    "subgroup_diagnostics": subgroup,
    "split": {
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    },
}
with open(f"{ART}/analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
df.to_csv(
    "./data/interim/cleaned_credit_risk_dataset.csv", sep=";", index=False, encoding="utf-8"
)
print(
    json.dumps(
        {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "duplicates": duplicates,
            "missing_total": int(df.isna().sum().sum()),
            "class_dist": class_dist,
            "best_model": best_name,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
)

# %%
