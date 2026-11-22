from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Data Avengers — Risco de Crédito",
    page_icon="💳",
    layout="wide",
)

pages = [
    st.Page(
        "pages/01_resultados.py",
        title="Resultados",
        icon="📊",
        default=True,
    ),
    st.Page(
        "pages/02_inferencia.py",
        title="Inferência",
        icon="🔮",
    ),
]

pg = st.navigation(pages)
pg.run()
