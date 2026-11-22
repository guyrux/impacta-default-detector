from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from default_detector.analysis import (
    build_analysis_payload,
    get_project_root,
    load_credit_dataset,
    save_analysis_payload,
)
from default_detector.charts import (
    plot_categorical_rates,
    plot_cv_summary,
    plot_metrics_radar,
    plot_numeric_correlation,
    plot_permutation_importance,
    plot_subgroup_diagnostics,
    plot_target_distribution,
)

ROOT = get_project_root()
JSON_PATH = ROOT / "artifacts" / "analysis_results.json"


def inject_css() -> None:
    css_path = ROOT / "static" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def header() -> None:
    st.markdown(
        """
        <div class="app-header">
          <span class="eyebrow">Projeto de dados &middot; Classifica&ccedil;&atilde;o bin&aacute;ria</span>
          <h1>Risco de cr&eacute;dito: quem vai deixar de pagar?</h1>
          <p class="subtitle">
            Da auditoria de qualidade &agrave; compara&ccedil;&atilde;o de modelos &mdash; Regress&atilde;o Logistica e
            Random Forest com valida&ccedil;&atilde;o cruzada estratificada, importa&ccedil;&atilde;o por permuta&ccedil;&atilde;o e
            diagn&oacute;stico por subgrupo.
          </p>
          <button class="btn-orange">Carregar meu analysis_results.json</button>
          <span class="header-hint">Exibindo resultados de refer&ecirc;ncia do credit_risk_dataset</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_payload() -> dict:
    """Garante que o payload JSON exista e seja válido, gerando-o quando ausente
    ou corrompido."""
    needs_generation = not JSON_PATH.exists()
    if not needs_generation:
        try:
            with JSON_PATH.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            needs_generation = True

    if needs_generation:
        with st.spinner("Gerando an&aacute;lise e payload JSON (pode demorar)..."):
            df = load_credit_dataset(ROOT)
            payload = build_analysis_payload(df, ROOT)
            save_analysis_payload(payload, ROOT, JSON_PATH)
        with JSON_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def kpi_grid(payload: dict) -> None:
    dataset = payload["dataset"]
    quality = payload["quality"]
    models_test = payload["models_test"]
    best_model = payload["best_model"]
    split = payload["split"]

    total_rows = dataset["rows"]
    n_cols = dataset["columns"]
    class_dist = payload["class_distribution"]
    total = sum(class_dist.values())
    pos_rate = (class_dist.get("1", 0) / total * 100) if total else 0.0
    n_positive = int(class_dist.get("1", 0))
    best_roc = models_test[best_model]["roc_auc"] if best_model in models_test else 0.0
    best_recall = models_test[best_model]["recall"] if best_model in models_test else 0.0
    test_rows = split["test_rows"]

    cards = [
        {
            "label": "REGISTROS",
            "value": f"{total_rows:,}".replace(",", "."),
            "sub": f"{n_cols} colunas",
        },
        {
            "label": "TAXA DE INADIMPLÊNCIA",
            "value": f"{pos_rate:.1f}%",
            "sub": f"{n_positive:,} eventos positivos".replace(",", "."),
        },
        {
            "label": "MELHOR MODELO",
            "value": f"{best_roc * 100:.1f}%",
            "sub": f"ROC-AUC &middot; {best_model}",
        },
        {
            "label": "RECALL NO TESTE",
            "value": f"{best_recall * 100:.1f}%",
            "sub": f"{test_rows:,} linhas de teste".replace(",", "."),
        },
    ]

    html = '<div class="kpi-grid">'
    for card in cards:
        html += (
            '<div class="kpi-card">'
            f'<div class="kpi-label">{card["label"]}</div>'
            f'<div class="kpi-value">{card["value"]}</div>'
            f'<p class="kpi-sub">{card["sub"]}</p>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_header(num: str, title: str, subtitle: str | None = None) -> None:
    sub_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section">
          <div class="section-header">
            <span class="section-number">{num}</span>
            <h2 class="section-title">{title}</h2>
          </div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
          <div style="font-weight:700; font-size:0.85rem; text-transform:uppercase;
                      letter-spacing:0.06em; color:var(--text-muted); margin-bottom:10px;">
            {title}
          </div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Resultados — Data Avengers", layout="wide")
    inject_css()

    payload = ensure_payload()

    header()
    kpi_grid(payload)

    # ------------------------------------------------------------------
    # 01 Qualidade dos dados
    # ------------------------------------------------------------------
    section_header("01", "Qualidade dos dados",
                   "Ausentes, duplicados, regras de valida&ccedil;&atilde;e e outliers pelo crit&eacute;rio IQR (1,5&times;).")

    quality = payload["quality"]
    invalid = quality["invalid"]
    col_na_total = sum(quality["missing"].values())

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        info_card("VALORES AUSENTES",
                  f'<div style="font-size:2.2rem; font-weight:800; color:var(--text-main);">{col_na_total}</div>'
                  '<div style="color:var(--text-muted); font-size:0.8rem;">total de c&eacute;lulas vazias</div>')
    with c2:
        regras_html = ""
        for k, v in invalid.items():
            nome = k.replace("_", " ").title()
            regras_html += f'<div style="display:flex; justify-content:space-between; font-size:0.85rem; margin:3px 0;">'
            regras_html += f'<span style="color:var(--text-muted)">{nome}</span>'
            regras_html += f'<span class="tag tag-ok">ok</span></div>'
        info_card("REGRAS DE VALIDAÇÃO", regras_html)
    with c3:
        outliers = payload["outliers_iqr"]
        total_out = sum(o["count"] for o in outliers.values())
        info_card("OUTLIERS (IQR) POR VARIÁVEL",
                  f'<div style="font-size:2.2rem; font-weight:800; color:var(--text-main);">{total_out}</div>'
                  '<div style="color:var(--text-muted); font-size:0.8rem;">registros fora do intervalo 1,5&times; IQR</div>')

    st.divider()

    # ------------------------------------------------------------------
    # 02 Análise exploratória
    # ------------------------------------------------------------------
    section_header("02", "An&aacute;lise explorat&oacute;ria",
                   "Como o alvo se distribui e quais vari&aacute;veis se movem com ele.")

    col_donut, col_corr = st.columns([1, 1.4])
    with col_donut:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("Distribui&ccedil;&atilde;o do alvo")
        fig = plot_target_distribution(payload["class_distribution"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_corr:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("Correla&ccedil;&atilde;o de Pearson com o alvo")
        fig = plot_numeric_correlation(payload["numeric_correlations_with_target"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Taxa de inadimpl&ecirc;ncia por categoria**")
    chips = [
        ("classifica&ccedil;&atilde;o do empr&eacute;stimo", "classifica&ccedil;&atilde;o do empr&eacute;stimo"),
        ("tipo de propriedade da resid&e^ncia", "tipo de propriedade da resid&e^ncia"),
        ("finalidade do empr&eacute;stimo", "finalidade do empr&eacute;stimo"),
        ("hist&oacute;rico de inadimpl&ecirc;ncia no cadastro de cr&eacute;dito", "hist&oacute;rico de inadimpl&ecirc;ncia no cadastro de cr&eacute;dito"),
    ]
    chip_html = ""
    for _, label in chips:
        chip_html += f'<span class="tag tag-chip">{label}</span>'
    st.markdown(f'<div style="margin-bottom:14px;">{chip_html}</div>', unsafe_allow_html=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    fig = plot_categorical_rates(payload["target_rates_by_category"])
    st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Vermelho = acima da taxa m&eacute;ria (21,8%).")

    st.divider()

    # ------------------------------------------------------------------
    # 03 Modelagem
    # ------------------------------------------------------------------
    section_header("03", "Modelagem",
                   "Split 80%/20% estratificado &middot; random_state=42 &middot; StratifiedKFold (5).")

    col_radar, col_cv = st.columns([1, 1.2])
    with col_radar:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("M&eacute;tricas no conjunto de teste")
        fig = plot_metrics_radar(payload["models_test"], payload["best_model"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_cv:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("Valida&ccedil;&atilde;o cruzada (m&eacute;dia &plusmn; desvio)")
        fig = plot_cv_summary(payload["models_cv"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tabela comparativa
    models_test = payload["models_test"]
    best_model = payload["best_model"]
    rows_html = ""
    metric_pt = {
        "accuracy": "Acur&aacute;cia",
        "precision": "Precis&atilde;o",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "average_precision": "AP",
    }
    for met_key, met_label in metric_pt.items():
        v_lr = models_test["Regressão Logística"][met_key]
        v_rf = models_test["Random Forest"][met_key]
        cls_lr = "font-weight:700; color:var(--text-main);" if best_model == "Regressão Logística" else ""
        cls_rf = "font-weight:700; color:var(--text-main);" if best_model == "Random Forest" else ""
        rows_html += (
            f"<tr>"
            f"<td style='padding:8px 6px; color:var(--text-muted); font-size:0.88rem;'>{met_label}</td>"
            f"<td style='padding:8px 6px; {cls_lr}'>{v_lr}</td>"
            f"<td style='padding:8px 6px; {cls_rf}'>{v_rf}</td>"
            f"</tr>"
        )
    st.markdown(
        f"""
        <div class="info-card">
          <div style="font-weight:700; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.06em;
                      color:var(--text-muted); margin-bottom:10px;">Tabela comparativa</div>
          <table style="width:100%; border-collapse:collapse;">
            <thead>
              <tr>
                <th style="text-align:left; padding:8px 6px; color:var(--text-muted); font-size:0.8rem;
                           text-transform:uppercase; letter-spacing:0.05em; border-bottom:1px solid var(--border);">M&eacute;trica</th>
                <th style="text-align:left; padding:8px 6px; color:var(--text-muted); font-size:0.8rem;
                           text-transform:uppercase; letter-spacing:0.05em; border-bottom:1px solid var(--border);">Regress&atilde;o Log&iacute;stica</th>
                <th style="text-align:left; padding:8px 6px; color:var(--text-muted); font-size:0.8rem;
                           text-transform:uppercase; letter-spacing:0.05em; border-bottom:1px solid var(--border);">Random Forest</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Matriz de confusão (melhor modelo)
    cm = models_test[best_model]["confusion_matrix"]
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("Matriz de confus&atilde;o")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    st.markdown(
        f"""
        <div style="display:flex; gap:12px; margin-top:12px;">
          <div style="flex:1; background:#cbd5e1; border-radius:8px; padding:16px; text-align:center;">
            <div style="font-size:0.8rem; color:var(--text-muted);">Real 0 &middot; Previsto 0</div>
            <div style="font-size:1.8rem; font-weight:800; color:var(--accent-teal-dark);">{tn:,}</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">VN</div>
          </div>
          <div style="flex:1; background:#fef2f2; border-radius:8px; padding:16px; text-align:center;">
            <div style="font-size:0.8rem; color:var(--text-muted);">Real 0 &middot; Previsto 1</div>
            <div style="font-size:1.8rem; font-weight:800; color:var(--accent-red);">{fp:,}</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">FP</div>
          </div>
        </div>
        <div style="display:flex; gap:12px; margin-top:8px;">
          <div style="flex:1; background:#fef2f2; border-radius:8px; padding:16px; text-align:center;">
            <div style="font-size:0.8rem; color:var(--text-muted);">Real 1 &middot; Previsto 0</div>
            <div style="font-size:1.8rem; font-weight:800; color:var(--accent-red);">{fn:,}</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">FN</div>
          </div>
          <div style="flex:1; background:#e0f2fe; border-radius:8px; padding:16px; text-align:center;">
            <div style="font-size:0.8rem; color:var(--text-muted);">Real 1 &middot; Previsto 1</div>
            <div style="font-size:1.8rem; font-weight:800; color:var(--accent-teal-dark);">{tp:,}</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">VP</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # 04 Explicabilidade e equidade
    # ------------------------------------------------------------------
    section_header("04", "Explicabilidade e equidade",
                   "Import&acirc;ncia por permuta&ccedil;&atilde;o (queda no ROC-AUC) e diagn&oacute;stico por subgrupo &mdash; Random Forest.")

    col_imp, col_sub = st.columns([1, 1.2])
    with col_imp:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("Top 10 vari&aacute;veis por importa&ccedil;&atilde;o")
        fig = plot_permutation_importance(payload["permutation_importance"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_sub:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("Diagn&oacute;stico por subgrupo")
        chip_html = ""
        for label in ["classifica&ccedil;&atilde;o do empr&eacute;stimo",
                      "tipo de propriedade da resid&e^ncia",
                      "hist&oacute;rico de inadimpl&ecirc;ncia no cadastro de cr&eacute;dito"]:
            chip_html += f'<span class="tag tag-chip">{label}</span>'
        st.markdown(f'<div style="margin-bottom:10px;">{chip_html}</div>', unsafe_allow_html=True)
        fig = plot_subgroup_diagnostics(payload["subgroup_diagnostics"])
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Recall desigual entre grupos indica onde o modelo deixa passar mais inadimplentes.")

    st.divider()

    # ------------------------------------------------------------------
    # 05 Leitura do cientista
    # ------------------------------------------------------------------
    section_header("05", "Leitura do cientista", "Conclus&otilde;es e pr&oacute;ximos passos.")

    col_sinal, col_modelo, col_passos = st.columns([1, 1, 1])
    with col_sinal:
        info_card("Sinal",
                  "Comprometimento de renda, taxa de juros e grade do empr&eacute;stimo concentram o risco. "
                  "Locat&aacute;rios (RENT) e grades D&ndash;G s&atilde;o os segmentos cr&iacute;ticos.")
    with col_modelo:
        info_card("Modelo",
                  "Random Forest lidera em ROC-AUC com baixa vari&acirc;ncia na CV &mdash; est&aacute;vel. "
                  "&Arvores captam intera&ccedil;&otilde;es n&atilde;o lineares que a log&iacute;stica perde.")
    with col_passos:
        info_card("Pr&oacute;ximos passos",
                  "Ajustar o limiar pelo custo de FN vs FP, calibrar probabilidades, "
                  "testar HistGradientBoosting (j&aacute; importado) e monitorar drift.")


if __name__ == "__main__":
    main()
