from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET_COLUMN = "status do empréstimo"


def _rocket_palette(n_colors: int = 2):
    return sns.color_palette("rocket", n_colors=max(2, n_colors))


def _resolve_save_path(
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path | None:
    if save_path is not None:
        path = Path(save_path)
    elif output_dir is not None and filename is not None:
        path = Path(output_dir) / filename
    else:
        path = None

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_plot(save_path: Path | None, dpi: int = 180) -> None:
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_target_distribution(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str = "01_target_distribution.png",
    dpi: int = 180,
) -> Path | None:
    path = _resolve_save_path(save_path, output_dir, filename)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        dodge=False,
        palette=_rocket_palette(df[TARGET_COLUMN].nunique()),
        ax=ax,
        legend=False,
    )

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = int(len(df))
    for patch, class_value in zip(ax.patches, counts.index.to_list()):
        percentage = (counts[class_value] / total * 100) if total else 0.0
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() * 0.5,
            f"{percentage:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Distribuição do status do empréstimo")
    ax.set_xlabel("Status do empréstimo (0 = classe 0; 1 = evento positivo)")
    ax.set_ylabel("Quantidade")
    fig.tight_layout()
    _finalize_plot(path, dpi=dpi)
    return path


def plot_numeric_by_target(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str = "02_numeric_by_target.png",
    dpi: int = 180,
    showfliers=True
) -> Path | None:
    path = _resolve_save_path(save_path, output_dir, filename)
    numeric_fields = [
        "renda",
        "valor do empréstimo",
        "taxa de juros do empréstimo",
        "Percentual da renda comprometido com o empréstimo",
    ]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    palette = _rocket_palette(df[TARGET_COLUMN].nunique())
    for ax, column in zip(axes.flat, numeric_fields):
        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=column,
            hue=TARGET_COLUMN,
            dodge=False,
            ax=ax,
            palette=palette,
            legend=False,
            showfliers=showfliers
        )
        ax.set_title(column)
        ax.set_xlabel("Status do empréstimo")
    fig.tight_layout()
    _finalize_plot(path, dpi=dpi)
    return path


def plot_numeric_correlation(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str = "03_correlation.png",
    dpi: int = 180,
) -> Path | None:
    path = _resolve_save_path(save_path, output_dir, filename)
    numeric_fields = [
        "idade",
        "renda",
        "tempo de emprego",
        "valor do empréstimo",
        "taxa de juros do empréstimo",
        "Percentual da renda comprometido com o empréstimo",
        "Tempo de histórico de crédito",
        TARGET_COLUMN,
    ]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        df[numeric_fields].corr(),
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("rocket", as_cmap=True),
        center=0,
        ax=ax,
    )
    ax.set_title("Correlação entre variáveis numéricas")
    fig.tight_layout()
    _finalize_plot(path, dpi=dpi)
    return path


def plot_categorical_rates(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str = "04_categorical_payment_rates.png",
    dpi: int = 180,
) -> Path | None:
    path = _resolve_save_path(save_path, output_dir, filename)
    categorical_fields = [
        "classificação do empréstimo",
        "tipo de propriedade da residência",
        "finalidade do empréstimo",
        "histórico de inadimplência no cadastro de crédito",
    ]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, column in zip(axes.flat, categorical_fields):
        rates = df.groupby(column)[TARGET_COLUMN].mean().sort_values(ascending=False)
        colors = _rocket_palette(len(rates))
        ax.barh(rates.index, rates.values, color=colors)
        ax.set_title("Taxa da classe positiva por " + column)
        ax.set_xlabel("Média de status do empréstimo")
        ax.set_ylabel("")
    fig.tight_layout()
    _finalize_plot(path, dpi=dpi)
    return path


def plot_roc_curves(
    curves: list[tuple[str, Any, Any, float]],
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str = "05_roc_curves.png",
    dpi: int = 180,
) -> Path | None:
    path = _resolve_save_path(save_path, output_dir, filename)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curvas ROC no conjunto de teste")
    ax.legend()
    fig.tight_layout()
    _finalize_plot(path, dpi=dpi)
    return path


def plot_permutation_importance(
    importance: list[dict[str, Any]],
    best_model_name: str,
    save_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str = "06_permutation_importance.png",
    dpi: int = 180,
) -> Path | None:
    path = _resolve_save_path(save_path, output_dir, filename)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance[:10][::-1]
    palette = _rocket_palette(len(top))
    ax.barh(
        [item["feature"] for item in top],
        [item["mean_decrease_auc"] for item in top],
        color=palette,
    )
    ax.set_xlabel("Queda média no ROC-AUC")
    ax.set_title("Importância por permutação — " + best_model_name)
    fig.tight_layout()
    _finalize_plot(path, dpi=dpi)
    return path
