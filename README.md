# 💳 Detecção de Fraude em Cartão de Crédito

> Identifica transações fraudulentas em tempo real usando Machine Learning.

Projeto Final — CC0121 (Inteligência Artificial) — UNIFAP — 2026.1

---

## Equipe

| Nome Completo | Matrícula | Papel no projeto |
|---|---|---|
| Michel Batista do Monte | 2019007467 | [A PREENCHER] |
| Thiago Lobato Rodrigues | 2021009558 | [A PREENCHER] |
| Welliton Nunes Almeida | 2019010560 | [A PREENCHER] |

---

## O Problema

Fraudes em transações de cartão de crédito causam prejuízos bilionários anuais a bancos, fintechs e consumidores. Com milhares de transações processadas por segundo, a revisão manual é inviável — e cada fraude não detectada é dinheiro perdido, enquanto cada transação legítima bloqueada por engano é um cliente frustrado. Este projeto entrega um classificador automatizado que ajuda instituições financeiras a sinalizar transações suspeitas em tempo real, com nível de confiança, para revisão ou bloqueio imediato.

---

## A Solução

Um pipeline de Machine Learning que recebe os atributos de uma transação e retorna se ela é **legítima** ou **fraudulenta**, junto com o nível de confiança da predição. A interação acontece por um app web (Streamlit): o usuário preenche os campos da transação (ou carrega um exemplo real do dataset com um clique) e recebe a classificação instantaneamente.

**[COLOCAR AQUI um screenshot ou GIF do app rodando — ex: `docs/demo.gif` ou `docs/screenshot.png`]**

```markdown
![Demo do app](docs/demo.gif)
```

---

## Técnicas Usadas

| Técnica | Onde é usada | Por quê |
|---|---|---|
| **Análise Exploratória de Dados (EDA)** | `notebooks/01_eda.ipynb` | Entender distribuição de classes, padrões em `Amount`/`Time` e correlações antes de modelar |
| **StandardScaler** | `notebooks/02_preprocessing.ipynb` | Normalizar `Amount` e `Time`, únicas features fora da escala do PCA original |
| **SMOTE (oversampling)** | `notebooks/03_modeling.ipynb` | Corrigir o forte desbalanceamento de classes (0,17% de fraude) aplicando oversampling sintético **apenas no treino**, evitando vazamento de dados |
| **Regressão Logística** | `notebooks/03_modeling.ipynb` | Baseline simples e interpretável para comparação |
| **Random Forest** | `notebooks/03_modeling.ipynb` | Modelo escolhido — melhor equilíbrio entre Precision e Recall em dados tabulares desbalanceados |
| **XGBoost** | `notebooks/03_modeling.ipynb` | Comparação com gradient boosting, forte em problemas tabulares |
| **Stratified K-Fold Cross-Validation** | `notebooks/03_modeling.ipynb` | Validar que o desempenho do modelo escolhido é estável, mantendo a proporção de classes em cada fold |
| **Streamlit** | `app/app.py` | Interface web simples para simular a detecção em tempo real |

---

## Resultados

Métrica mais relevante para este problema: **F1-Score da classe fraude** (equilíbrio entre Precision e Recall), já que acurácia sozinha é enganosa com 99,83% dos dados sendo de uma única classe. AUC-ROC mede a separação geral entre as classes.

| Modelo | AUC-ROC | F1 (fraude) | Precision | Recall |
|---|---|---|---|---|
| **Random Forest ✅ (escolhido)** | 0,9754 | **0,8497** | 0,8632 | 0,8367 |
| XGBoost | 0,9808 | 0,7870 | 0,7203 | 0,8673 |
| Regressão Logística (baseline) | 0,9699 | 0,1092 | 0,0580 | 0,9184 |

**Meta da proposta:** AUC-ROC ≥ 0,95 **e** F1-Score (fraude) ≥ 0,80 → ✅ **atingida** com Random Forest.

O Random Forest foi escolhido por ser o único modelo que atinge as duas metas simultaneamente: identifica 84% das fraudes reais (Recall) com apenas 14% de falsos positivos entre os alertas gerados (Precision de 86%).

---

## Como Rodar (testado em máquina limpa)

### 1. Clonar o repositório

```bash
git clone https://github.com/lobat00/Deteccao-fraude-de-cartao-de-credito-projeto-CC0121.git
cd Deteccao-fraude-de-cartao-de-credito-projeto-CC0121
```

### 2. Criar e ativar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar credenciais do Kaggle

Gere um token em [kaggle.com](https://www.kaggle.com) → Settings → API, depois:

```bash
mkdir -p ~/.kaggle
echo "SEU_TOKEN_AQUI" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

### 5. Baixar o dataset

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
cd data && unzip creditcardfraud.zip && cd ..
```

### 6. Rodar os notebooks, nesta ordem

Abra no VS Code (extensão Jupyter), selecione o kernel da `venv` e rode célula por célula:

```
notebooks/01_eda.ipynb              # análise exploratória
notebooks/02_preprocessing.ipynb    # gera data/processed/ e models/scaler.pkl
notebooks/03_modeling.ipynb         # treina os modelos e gera models/best_model.pkl
```

### 7. Rodar o app

```bash
streamlit run app/app.py
```

Abre automaticamente em `http://localhost:8501`. Use os botões "Carregar exemplo de FRAUDE/LEGÍTIMO" para testar rapidamente com transações reais do dataset.

---

## Uso de IA

[A PREENCHER pela equipe — descrever com transparência como LLMs foram usados. Sugestão de estrutura abaixo, editar conforme o uso real de cada integrante:]

- **Ferramenta utilizada:** Claude (Anthropic)
- **Onde foi usada:**
  - Estruturação inicial do projeto (organização de pastas, `requirements.txt`, `.gitignore`)
  - Geração do esqueleto dos notebooks (`01_eda`, `02_preprocessing`, `03_modeling`) como ponto de partida, revisado e ajustado pela equipe
  - Geração inicial do app Streamlit (`app/app.py`)
  - Debug de erros específicos (ex: incompatibilidade de ordem de colunas no `predict()`, configuração de credenciais do Kaggle, problemas de `.gitignore`/tamanho de arquivo no GitHub)
  - Redação e organização deste README
- **O que a equipe decidiu por conta própria:** escolha do dataset, escolha entre os modelos comparados (Random Forest sobre XGBoost, priorizando F1 sobre AUC-ROC marginal), estratégia de balanceamento (SMOTE 50/50), interpretação dos resultados e validação das métricas contra a meta da proposta.
- **O que não foi usado:** [descrever, se aplicável — ex: geração de texto do relatório final sem revisão, etc.]

---

## Estrutura do Repositório

```
credit-card-fraud-detection/
├── data/                # dataset (não versionado — baixar via Kaggle API)
│   └── processed/       # dados pré-processados (gerado pelo notebook 02)
├── notebooks/           # 01_eda, 02_preprocessing, 03_modeling
├── models/               # scaler.pkl e best_model.pkl (gerados pelos notebooks)
├── app/                 # aplicação Streamlit
├── reports/              # relatório técnico final
├── docs/                  # screenshots/GIFs de demonstração
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Entregáveis

- [x] Repositório GitHub com código, notebooks e README
- [ ] Relatório técnico (4-8 páginas)
- [x] Aplicativo Streamlit funcional

## Datas

- Entrega da Proposta: 28/05/2026
- Apresentação Final: 16/07/2026