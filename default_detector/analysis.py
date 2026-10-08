from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLUMNS = [
    "idade",
    "renda",
    "tempo de emprego",
    "valor do empréstimo",
    "taxa de juros do empréstimo",
    "Percentual da renda comprometido com o empréstimo",
    "Tempo de histórico de crédito",
]

CAT_COLUMNS = [
    "tipo de propriedade da residência",
    "finalidade do empréstimo",
    "classificação do empréstimo",
    "histórico de inadimplência no cadastro de crédito",
]

TARGET_COLUMN = "status do empréstimo"


def get_project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[1]


def load_credit_dataset(
    project_root: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    root = get_project_root(project_root)
    file_path = (
        Path(csv_path)
        if csv_path is not None
        else root / "data" / "raw" / "credit_risk_dataset.csv"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    return pd.read_csv(file_path, sep=";", encoding="utf-8-sig")


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    required = NUM_COLUMNS + CAT_COLUMNS + [TARGET_COLUMN]
    return [column for column in required if column not in df.columns]


def summarize_quality(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    invalid = {
        "idade_nao_positiva": int((df["idade"] <= 0).sum()),
        "renda_nao_positiva": int((df["renda"] <= 0).sum()),
        "tempo_emprego_negativo": int((df["tempo de emprego"] < 0).sum()),
        "valor_emprestimo_nao_positivo": int((df["valor do empréstimo"] <= 0).sum()),
        "taxa_juros_nao_positiva": int((df["taxa de juros do empréstimo"] <= 0).sum()),
        "percentual_renda_emprestimo_fora_0_1": int(
            (
                (df["Percentual da renda comprometido com o empréstimo"] < 0)
                | (df["Percentual da renda comprometido com o empréstimo"] > 1)
            ).sum()
        ),
        "historico_credito_negativo": int(
            (df["Tempo de histórico de crédito"] < 0).sum()
        ),
        "status_emprestimo_nao_binario": int((~df[TARGET_COLUMN].isin([0, 1])).sum()),
    }
    return {
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def summarize_descriptive(df: pd.DataFrame) -> dict[str, Any]:
    summary = df[NUM_COLUMNS].describe().round(4).to_dict()
    cardinality = {
        column: sorted([str(value) for value in df[column].dropna().unique().tolist()])
        for column in CAT_COLUMNS + [TARGET_COLUMN]
    }
    class_distribution = df[TARGET_COLUMN].value_counts().sort_index().to_dict()

    outliers: dict[str, dict[str, Any]] = {}
    for column in NUM_COLUMNS:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
        outliers[column] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 2),
            "lower": float(lower),
            "upper": float(upper),
        }

    rates_by_cat: dict[str, list[dict[str, Any]]] = {}
    for column in CAT_COLUMNS:
        table = df.groupby(column)[TARGET_COLUMN].agg(["count", "mean"]).round(4)
        rates_by_cat[column] = table.reset_index().to_dict(orient="records")

    corr = (
        df[NUM_COLUMNS + [TARGET_COLUMN]]
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .round(4)
        .to_dict()
    )

    return {
        "summary": summary,
        "cardinality": cardinality,
        "class_distribution": {
            str(key): int(value) for key, value in class_distribution.items()
        },
        "outliers_iqr": outliers,
        "target_rates_by_category": rates_by_cat,
        "numeric_correlations_with_target": corr,
    }


def train_and_evaluate_models(df: pd.DataFrame) -> dict[str, Any]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUM_COLUMNS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLUMNS,
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

    results: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}
    curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    for name, estimator in models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "average_precision": round(float(average_precision_score(y_test, prob)), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(
                y_test, pred, output_dict=True
            ),
        }

        fitted_models[name] = pipeline
        fpr, tpr, _ = roc_curve(y_test, prob)
        curves.append((name, fpr, tpr, float(results[name]["roc_auc"])))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary: dict[str, Any] = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "f1", "recall", "precision"],
            n_jobs=-1,
        )
        cv_summary[name] = {
            metric: {
                "mean": round(float(scores[f"test_{metric}"].mean()), 4),
                "std": round(float(scores[f"test_{metric}"].std()), 4),
            }
            for metric in ["roc_auc", "f1", "recall", "precision"]
        }

    best_name = max(results, key=lambda item: results[item]["roc_auc"])
    best_model = fitted_models[best_name]
    permutation = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importance = sorted(
        [
            {
                "feature": feature,
                "mean_decrease_auc": round(float(value), 5),
                "std": round(float(std), 5),
            }
            for feature, value, std in zip(
                X.columns, permutation.importances_mean, permutation.importances_std
            )
        ],
        key=lambda item: item["mean_decrease_auc"],
        reverse=True,
    )

    subgroup_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for column in [
        "histórico de inadimplência no cadastro de crédito",
        "classificação do empréstimo",
        "tipo de propriedade da residência",
    ]:
        rows: list[dict[str, Any]] = []
        for level, group in df.loc[X_test.index].assign(_y=y_test).groupby(column):
            idx = group.index
            pred = best_model.predict(X.loc[idx])
            actual = y.loc[idx]
            rows.append(
                {
                    "group": str(level),
                    "n": int(len(idx)),
                    "positive_rate_actual": round(float(actual.mean()), 4),
                    "positive_rate_predicted": round(float(pred.mean()), 4),
                    "recall": round(
                        float(recall_score(actual, pred, zero_division=0)), 4
                    ),
                }
            )
        subgroup_diagnostics[column] = rows

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_results": results,
        "cv_summary": cv_summary,
        "best_model": best_name,
        "permutation_importance": importance,
        "subgroup_diagnostics": subgroup_diagnostics,
        "curves": curves,
    }


def build_analysis_payload(
    df: pd.DataFrame | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    source_df = df if df is not None else load_credit_dataset(project_root)
    quality = summarize_quality(source_df)
    descriptive = summarize_descriptive(source_df)
    modeling = train_and_evaluate_models(source_df)

    payload = {
        "dataset": {
            "rows": int(source_df.shape[0]),
            "columns": int(source_df.shape[1]),
            "original_columns": list(source_df.columns),
            "column_names": list(source_df.columns),
            "dtypes": {
                column: str(dtype) for column, dtype in source_df.dtypes.items()
            },
        },
        "quality": quality,
        "descriptive_statistics": descriptive["summary"],
        "cardinality": descriptive["cardinality"],
        "class_distribution": descriptive["class_distribution"],
        "outliers_iqr": descriptive["outliers_iqr"],
        "target_rates_by_category": descriptive["target_rates_by_category"],
        "numeric_correlations_with_target": descriptive[
            "numeric_correlations_with_target"
        ],
        "models_test": modeling["model_results"],
        "models_cv": modeling["cv_summary"],
        "best_model": modeling["best_model"],
        "permutation_importance": modeling["permutation_importance"],
        "subgroup_diagnostics": modeling["subgroup_diagnostics"],
        "curves": modeling["curves"],
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": int(modeling["X_train"].shape[0]),
            "test_rows": int(modeling["X_test"].shape[0]),
        },
    }
    return payload


def save_analysis_payload(
    payload: dict[str, Any],
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    root = get_project_root(project_root)
    final_path = (
        Path(output_path)
        if output_path is not None
        else root / "artifacts" / "analysis_results.json"
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return final_path
