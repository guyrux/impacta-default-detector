# AGENTS.md

Guia para agentes (de código, análise ou documentação) que trabalham neste repositório.

## Visão geral

Projeto acadêmico (Faculdade Impacta, disciplina Data Modeling) de análise de risco de crédito.
Explora o Credit Risk Dataset (Kaggle) e compara modelos de classificação para o alvo
`status do empréstimo`.

Idioma: português (pt-BR) em textos, documentação, rótulos de gráficos, mensagens de erro e
nomes de colunas; nomes de funções, variáveis e chaves de JSON em inglês.

## Metodologia

O projeto segue o **CRISP-ML(Q)** (Cross-Industry Standard Process for Machine Learning with
Quality Assurance). As fases são:

1. Business and Data Understanding
2. Data Preparation
3. Modeling
4. Evaluation
5. Deployment
6. Monitoring and Maintenance

Cada notebook corresponde a uma fase ou etapa da metodologia e é nomeado
`NN - <fase_em_snake_case>.ipynb` (ex.: `01 - business_and_data_understanding.ipynb`).
Novo trabalho deve ser colocado na fase adequada, com garantia de qualidade (validações,
reprodutibilidade, métricas documentadas) em cada etapa.

## Ambiente

- Python `>=3.12,<3.13`, gerenciado com **Poetry** (inclusive no VS Code).
- O pacote `default_detector` é instalado no ambiente também em desenvolvimento:
  `poetry install` (sem `--no-root`).
- Sempre execute comandos com `poetry run ...`.
- Novas dependências: `poetry add <pacote>` (dependências de desenvolvimento com
  `--group dev`). Mantenha `poetry.lock` versionado e não use `pip install` direto.

## Importações

- Importe sempre pelo pacote instalado: `from default_detector.analysis import ...`,
  `from default_detector import plots`.
- Não manipule `sys.path` nem dependa do diretório atual para encontrar o pacote, seja em
  scripts, notebooks ou testes.
- Módulos do pacote devem ser seguros para importação: nada de análise, leitura de dados ou
  gravação de arquivos no nível do módulo. Pontos de entrada ficam em `main()`.

## Estrutura

- `default_detector/analysis.py`: carga, validação, qualidade, estatísticas descritivas,
  modelagem e geração/gravação do payload JSON. Fonte da verdade para as constantes de colunas
  (`NUM_COLUMNS`, `CAT_COLUMNS`, `TARGET_COLUMN`).
- `default_detector/plots.py`: funções de gráficos (paleta seaborn `rocket`, tema `whitegrid`).
- `analyze_python.py`: shim na raiz que chama `default_detector.analyze_python.main()`;
  preferir executar com `poetry run python -m default_detector.analyze_python`. Duplica a
  lógica do pacote; ao alterar regras de análise, mantenha os dois coerentes.
- `notebooks/`: notebooks das fases do CRISP-ML(Q). Eles importam do pacote, em vez de
  redefinir a lógica.
- `tests/`: testes com pytest.
- `data/raw/`: dados de entrada (`credit_risk_dataset.csv`); `data/interim/`: saídas
  intermediárias.
- `artifacts/analysis_results.json` e `figures/*.png`: saídas geradas e versionadas.

## Dados

- CSV com separador `;`, codificação UTF-8 (com ou sem BOM). Leia com
  `sep=";", encoding="utf-8-sig"`.
- Cabeçalhos em português e com acentos (ex.: `"valor do empréstimo"`,
  `"Percentual da renda comprometido com o empréstimo"`). Não renomeie nem normalize colunas;
  use as constantes de `analysis.py`.
- Alvo binário `status do empréstimo`; a classe `1` é a positiva.
- Arquivos `data/*/*.csv` são ignorados pelo git. Nunca faça commit de dados.

## Convenções de código

- Use `from __future__ import annotations`, type hints e `pathlib.Path`.
- Resolva caminhos a partir da raiz do projeto (`get_project_root`), e não do diretório atual.
- Funções de gráfico seguem a mesma assinatura:
  `(dados, save_path=None, output_dir=None, filename="NN_nome.png", dpi=180) -> Path | None`
  e terminam com `_finalize_plot`. As figuras são numeradas (`01_`, `02_`, ...).
- Reprodutibilidade: `random_state=42`, divisão treino/teste estratificada 80/20 e
  `StratifiedKFold(n_splits=5, shuffle=True)`. Não altere sem motivo explícito.
- Todo o pré-processamento fica dentro de `Pipeline`/`ColumnTransformer`, para evitar
  vazamento de dados. A imputação dos modelos não é aplicada ao CSV exportado em
  `data/interim/`.
- Métricas no JSON são arredondadas (4 casas; importância por permutação com 5) e convertidas
  para tipos nativos (`float`/`int`) antes da serialização. O JSON é gravado com
  `ensure_ascii=False, indent=2`.
- Não altere chaves existentes do payload JSON sem atualizar quem as consome (notebooks,
  README).

## Testes

- Executar: `poetry run pytest`.
- Testes de gráficos usam o backend `Agg` (`matplotlib.use("Agg")`).

## Cuidados

- Regenerar `figures/` e `artifacts/` altera arquivos versionados. Só faça isso quando for
  intencional.
- Mensagens de commit seguem Conventional Commits em inglês (`feat:`, `chore:`, ...).
