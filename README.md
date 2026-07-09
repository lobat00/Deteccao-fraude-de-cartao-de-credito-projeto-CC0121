# Detecção de Fraude em Transações de Cartão de Crédito com Machine Learning

Projeto Final — CC0121 (Inteligência Artificial) — UNIFAP — 2026.1

## Equipe

| Nome Completo | Matrícula |
|---|---|
| Michel Batista do Monte | 2019007467 |
| Thiago Lobato Rodrigues | 2021009558 |
| Welliton Nunes Almeida | 2019010560 |

## Problema

Fraudes em transações de cartão de crédito representam um prejuízo bilionário anual para instituições financeiras e consumidores. A detecção manual é inviável dado o volume de transações processadas por segundo. Este projeto constrói um classificador automatizado capaz de identificar transações fraudulentas, beneficiando bancos, fintechs e clientes.

## Dataset

**Credit Card Fraud Detection** (Kaggle: `mlg-ulb/creditcardfraud`)

- 284.807 transações reais de cartões europeus (setembro de 2013)
- 492 transações fraudulentas (0,17% — forte desbalanceamento)
- 30 features numéricas (28 anonimizadas via PCA + `Time` + `Amount`)
- Variável alvo: `Class` (0 = legítima, 1 = fraude)

## Abordagem Técnica

- **EDA**: Pandas, Matplotlib, Seaborn
- **Balanceamento**: SMOTE (aplicado apenas no conjunto de treino)
- **Modelos**: Regressão Logística (baseline), Random Forest, XGBoost
- **Validação**: Stratified K-Fold — Precision, Recall, F1, AUC-ROC
- **Interface**: app Streamlit para simulação em tempo real
- **Ambiente**: desenvolvimento local (VS Code) + Google Colab

## Estrutura do Repositório

```
credit-card-fraud-detection/
├── data/               # dataset (não versionado)
├── notebooks/          # EDA e experimentos
├── src/                # código do pipeline
├── app/                # aplicação Streamlit
├── reports/            # relatório técnico final
├── requirements.txt
├── .gitignore
└── README.md
```

## Como Rodar

```bash
# Ambiente virtual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Baixar dataset (requer credenciais Kaggle configuradas)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
cd data && unzip creditcardfraud.zip && cd ..

# Rodar o app
streamlit run app/app.py
```

## Métricas de Sucesso

- AUC-ROC ≥ 0,95
- F1-Score (classe fraude) ≥ 0,80

## Entregáveis

- Repositório GitHub com código, notebooks e README
- Relatório técnico (3–5 páginas)
- Aplicativo Streamlit funcional

## Datas

- Entrega da Proposta: 28/05/2026
- Apresentação Final: 09/07/2026
