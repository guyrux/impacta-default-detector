# Data Avengers | Análise de Risco de Crédito

Projeto acadêmico desenvolvido para a disciplina **Data Modeling**, do curso **Tecnologia em Banco de Dados** da **Faculdade Impacta**.

A equipe Data Avengers explora dados de crédito e compara modelos de classificação para analisar a variável `loan_status`.

## Fonte dos dados

O conjunto de dados tem como referência o [Credit Risk Dataset no Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data).

Para executar a análise, disponibilize o arquivo preparado em:

```text
data/raw/credit_risk_dataset.csv
```

O script espera um CSV separado por ponto e vírgula (`;`), codificado em UTF-8 (com ou sem BOM), com os cabeçalhos em português reconhecidos pelo código. Ele renomeia esses campos para nomes em inglês durante a análise. O arquivo disponível neste projeto serve como referência do formato esperado.

## Requisitos

- Python `>=3.12,<3.13`
- Poetry

## Como executar

Na raiz do projeto, instale as dependências:

```bash
poetry install --no-root
```

Execute a análise:

```bash
poetry run python analyze_python.py
```

## Análises realizadas

- Qualidade dos dados: valores ausentes, duplicatas e verificações de valores inválidos.
- Estatísticas descritivas, cardinalidade, distribuição das classes e outliers pelo método IQR.
- Relações entre o alvo e variáveis categóricas, além de correlações numéricas.
- Pré-processamento em pipelines: imputação e padronização das variáveis numéricas; imputação e codificação one-hot das categóricas.
- Comparação entre Regressão Logística e Random Forest, com divisão estratificada de treino e teste (80/20) e validação cruzada estratificada de cinco folds no conjunto de treino.
- Avaliação por acurácia, precisão, recall, F1, ROC-AUC e average precision; inclui matriz de confusão e relatório de classificação.
- Importância por permutação do modelo com maior ROC-AUC no conjunto de teste e diagnósticos de subgrupos.

A classe `1` é tratada pelo código como a classe positiva. A interpretação de negócio de cada valor de `loan_status` deve seguir a documentação da fonte dos dados.

## Arquivos gerados

- `artifacts/analysis_results.json`: métricas, estatísticas, verificações de qualidade, importância das variáveis e diagnósticos por subgrupo.
- `data/interim/cleaned_credit_risk_dataset.csv`: CSV exportado com os nomes das colunas padronizados. A imputação usada nos modelos não é aplicada a essa exportação.
- `figures/01_target_distribution.png`: distribuição do alvo.
- `figures/02_numeric_by_target.png`: variáveis numéricas por classe.
- `figures/03_correlation.png`: correlação entre variáveis numéricas e o alvo.
- `figures/04_categorical_payment_rates.png`: média do alvo por categoria.
- `figures/05_roc_curves.png`: curvas ROC no conjunto de teste.
- `figures/06_permutation_importance.png`: importância por permutação.

## Estrutura do projeto

```text
.
├── analyze_python.py
├── artifacts/
│   └── analysis_results.json
├── data/
│   ├── interim/
│   │   └── cleaned_credit_risk_dataset.csv
│   └── raw/
│       └── credit_risk_dataset.csv
├── figures/
├── poetry.lock
└── pyproject.toml
```
