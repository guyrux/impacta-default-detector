from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# Paleta de cores do layout de referência (teal, laranja, vermelho).
TEAL = "#0f766e"
TEAL_DARK = "#0b5f56"
ORANGE = "#f59e0b"
ORANGE_DARK = "#d97706"
RED = "#dc2626"
GRAY = "#9ca3af"
BG = "#f7f3ee"


def _base_layout(title: str, height: int = 420) -> go.Layout:
    return go.Layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color="#111827")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#111827"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=False),
    )


def plot_target_distribution(class_distribution: dict[str, int]) -> go.Figure:
    """Gráfico de rosca (donut) da distribuição do alvo."""
    labels = [f"Classe {k}" for k in sorted(class_distribution, key=int)]
    values = [int(class_distribution[k]) for k in sorted(class_distribution, key=int)]
    colors = [TEAL, RED]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(_base_layout("Distribuição do status do empréstimo", height=380))
    fig.update_layout(showlegend=False)
    return fig


def plot_numeric_correlation(corr: dict[str, float]) -> go.Figure:
    """Barras horizontais das correlações de Pearson com o alvo."""
    features = list(corr.keys())
    values = [float(v) for v in corr.values()]
    # Cor diverge conforme o sinal da correlação.
    colors = [TEAL if v >= 0 else RED for v in values]
    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker=dict(color=colors),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        _base_layout("Correlação de Pearson com o alvo", height=420),
        xaxis=dict(title="Correlação", zeroline=True, zerolinecolor=GRAY),
        yaxis=dict(title="", autorange="reversed"),
    )
    return fig


def plot_categorical_rates(
    target_rates_by_category: dict[str, list[dict[str, Any]]],
) -> go.Figure:
    """Barras horizontais da taxa da classe positiva por categoria."""
    # Pega a primeira variável categórica para exibir (pode ser estendido com seletor).
    column = list(target_rates_by_category.keys())[0]
    rows = target_rates_by_category[column]
    cats = [str(r[column]) for r in rows]
    rates = [float(r["mean"]) * 100 for r in rows]
    order = sorted(range(len(rates)), key=lambda i: rates[i], reverse=True)
    cats = [cats[i] for i in order]
    rates = [rates[i] for i in order]
    fig = go.Figure(
        data=[
            go.Bar(
                x=rates,
                y=cats,
                orientation="h",
                marker=dict(color=ORANGE),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        _base_layout(f"Taxa da classe positiva por {column}", height=420),
        xaxis=dict(title="Taxa da classe positiva (%)", range=[0, max(rates) * 1.15 if rates else 100]),
        yaxis=dict(title=""),
    )
    return fig


def plot_metrics_radar(models_test: dict[str, Any], best_model: str) -> go.Figure:
    """Radar chart comparando métricas dos modelos."""
    metrics = ["Acurácia", "Precisão", "Recall", "F1", "ROC-AUC", "AP"]
    key_map = {
        "Acurácia": "accuracy",
        "Precisão": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
        "AP": "average_precision",
    }
    fig = go.Figure()
    model_colors = {"Regressão Logística": TEAL, "Random Forest": ORANGE}
    for name, metrics_dict in models_test.items():
        values = [float(metrics_dict[key_map[m]]) for m in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics,
                name=name,
                line=dict(color=model_colors.get(name, GRAY), width=3),
                fill="toself",
                fillcolor=model_colors.get(name, GRAY) + "22",
                hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        _base_layout("Métricas no conjunto de teste", height=420),
        polar=dict(
            radialaxis=dict(range=[0, 1], showgrid=True, gridcolor="#e5e7eb"),
            angularaxis=dict(direction="clockwise"),
        ),
        legend=dict(title_text="Modelo"),
    )
    # Destaca o melhor modelo na legenda (via ordem / cor já é suficiente).
    return fig


def plot_cv_summary(models_cv: dict[str, Any]) -> go.Figure:
    """Barras agrupadas da validação cruzada (média ± desvio)."""
    metrics = ["roc_auc", "f1", "recall", "precision"]
    model_names = list(models_cv.keys())
    model_colors = {"Regressão Logística": TEAL, "Random Forest": ORANGE}
    fig = go.Figure()
    for name in model_names:
        means = [float(models_cv[name][m]["mean"]) for m in metrics]
        stds = [float(models_cv[name][m]["std"]) for m in metrics]
        fig.add_trace(
            go.Bar(
                name=name,
                x=[m.upper() for m in metrics],
                y=means,
                error_y=dict(type="data", array=stds, visible=True, thickness=1.5, width=3),
                marker=dict(color=model_colors.get(name, GRAY)),
                hovertemplate="%{x}: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        _base_layout("Validação cruzada (média ± desvio)", height=420),
        barmode="group",
        xaxis=dict(title="Métrica"),
        yaxis=dict(title="Valor", range=[0, 1]),
        legend=dict(title_text="Modelo"),
    )
    return fig


def plot_permutation_importance(
    importance: list[dict[str, Any]], top_n: int = 10
) -> go.Figure:
    """Barras horizontais da importância por permutação."""
    top = importance[:top_n][::-1]
    features = [item["feature"] for item in top]
    values = [float(item["mean_decrease_auc"]) for item in top]
    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker=dict(color=TEAL),
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        _base_layout("Importância por permutação (queda no ROC-AUC)", height=420),
        xaxis=dict(title="Queda média no ROC-AUC"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def plot_subgroup_diagnostics(
    subgroup_diagnostics: dict[str, list[dict[str, Any]]],
) -> go.Figure:
    """Barras agrupadas do diagnóstico por subgrupo (taxa real, prevista, recall)."""
    column = list(subgroup_diagnostics.keys())[0]
    rows = subgroup_diagnostics[column]
    groups = [str(r["group"]) for r in rows]
    real = [float(r["positive_rate_actual"]) for r in rows]
    prevista = [float(r["positive_rate_predicted"]) for r in rows]
    recall = [float(r["recall"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Taxa real",
            x=groups,
            y=real,
            marker=dict(color=TEAL),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Taxa prevista",
            x=groups,
            y=prevista,
            marker=dict(color=ORANGE),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Recall",
            x=groups,
            y=recall,
            marker=dict(color=RED),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        _base_layout(f"Diagnóstico por subgrupo — {column}", height=420),
        barmode="group",
        xaxis=dict(title="Grupo"),
        yaxis=dict(title="Taxa", tickformat=".0%"),
        legend=dict(title_text="Métrica"),
    )
    return fig
