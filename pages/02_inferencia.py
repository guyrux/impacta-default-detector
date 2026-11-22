from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from default_detector.analysis import (
    CAT_COLUMNS,
    NUM_COLUMNS,
    TARGET_COLUMN,
    ensure_id_column,
    get_model_dir,
    get_project_root,
    load_credit_dataset,
    load_model,
    save_model,
    train_best_pipeline,
)

ROOT = get_project_root()
MODEL_PATH = get_model_dir(ROOT) / "best_model.joblib"


def inject_css() -> None:
    css_path = ROOT / "static" / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def header() -> None:
    st.markdown(
        """
        <div class="app-header">
          <span class="eyebrow">Projeto de dados &middot; Infer&ecirc;ncia</span>
          <h1>Infer&ecirc;ncia a partir de um registro</h1>
          <p class="subtitle">
            Selecione um registro da base por ID e obtenha a predi&ccedil;&atilde;o de
            `status do empr&eacute;stimo` (classe 1 = evento positivo), usando o modelo salvo em
            <code>models/best_model.joblib</code>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_model() -> None:
    """Garante que o modelo joblib exista, treinando-o quando ausente."""
    if not MODEL_PATH.exists():
        with st.spinner(
            "Treinando o melhor modelo e salvando em ./models/ (pode demorar)..."
        ):
            df = load_credit_dataset(ROOT)
            pipeline = train_best_pipeline(df)
            save_model(pipeline, ROOT)


def load_data_with_id() -> pd.DataFrame:
    df = load_credit_dataset(ROOT)
    return ensure_id_column(df)


def predict_record(pipeline, record: pd.DataFrame) -> tuple[int, float]:
    """Retorna (classe prevista, probabilidade da classe positiva)."""
    features = record.drop(columns=["id"], errors="ignore")
    prediction = int(pipeline.predict(features)[0])
    probability = float(pipeline.predict_proba(features)[:, 1][0])
    return prediction, probability


def main() -> None:
    st.set_page_config(page_title="Inferência — Data Avengers", layout="wide")
    inject_css()

    header()

    ensure_model()

    df = load_data_with_id()

    # ---------------------------------------------------------------
    # Seleção do registro
    # ---------------------------------------------------------------
    st.markdown(
        """
        <div class="section">
          <div class="section-header">
            <span class="section-number"></span>
            <h2 class="section-title">Selecionar registro</h2>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    id_options = df["id"].tolist()
    selected_id = st.selectbox(
        "ID do registro",
        options=id_options,
        format_func=lambda i: f"ID {i}",
        label_visibility="collapsed",
    )

    record = df[df["id"] == selected_id]
    if record.empty:
        st.error("Registro n&atilde;o encontrado.")
        return

    record = record.iloc[0]
    pipeline = load_model()
    model_file_mtime = MODEL_PATH.stat().st_mtime
    st.caption(f"Modelo carregado de: `{MODEL_PATH}`")

    col_registry, col_prediction = st.columns(2)
    with col_registry:
        st.markdown("**Valores do registro selecionado**")
        display_cols = [c for c in df.columns if c not in (TARGET_COLUMN, "id")]
        prediction, probability = predict_record(
            pipeline,
            pd.DataFrame([record])[display_cols],
        )
        display_df = pd.DataFrame({coluna: [record[coluna]] for coluna in display_cols}).T
        display_df.columns = ["Valor"]
        st.dataframe(display_df, width="stretch")

    with col_prediction:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="inference-box danger">
                    <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;
                                color:var(--text-muted); margin-bottom:6px;">Predi&ccedil;&atilde;o</div>
                    <div>🚨 Classe 1 (risco de inadimpl&ecirc;ncia / evento positivo)</div>
                    <div class="big-prob">{probability:.1%}</div>
                    <div style="font-size:0.8rem; color:var(--text-muted); margin-top:6px;">Probabilidade da classe positiva</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="inference-box success">
                    <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;
                                color:var(--text-muted); margin-bottom:6px;">Predi&ccedil;&atilde;o</div>
                    <div>✅ Classe 0 (sem evento / baixo risco)</div>
                    <div class="big-prob">{probability:.1%}</div>
                    <div style="font-size:0.8rem; color:var(--text-muted); margin-top:6px;">Probabilidade da classe positiva</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    

    actual = record[TARGET_COLUMN]
    st.info(f"Valor real de `{TARGET_COLUMN}` na base: **{int(actual)}**")

    st.divider()

    # ---------------------------------------------------------------
    # Predição
    # ---------------------------------------------------------------
    st.markdown(
        """
        <div class="section">
          <div class="section-header">
            <span class="section-number"></span>
            <h2 class="section-title">Predi&ccedil;&atilde;o do modelo</h2>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.metric(
            label="Probabilidade da classe positiva (1)",
            value=f"{probability:.1%}",
        )
    with col_neg:
        st.metric(
            label="Probabilidade da classe negativa (0)",
            value=f"{1 - probability:.1%}",
        )

    st.divider()

    with st.expander("Detalhes t&eacute;cnicos"):
        st.write("Colunas num&eacute;ricas usadas pelo modelo:", NUM_COLUMNS)
        st.write("Colunas categ&oacute;ricas usadas pelo modelo:", CAT_COLUMNS)
        st.write("Target:", TARGET_COLUMN)
        st.write("Arquivo do modelo:", str(MODEL_PATH))
        st.write(
            "Última modifica&ccedil;&atilde;o do modelo:",
            pd.Timestamp(model_file_mtime, unit="s"),
        )


if __name__ == "__main__":
    main()
