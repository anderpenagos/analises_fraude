# Análise de Fraude com Machine Learning

Detecção de Transações Bancárias Fraudulentas

Este repositório apresenta um estudo prático de detecção de fraude utilizando um dataset de transações bancárias rotuladas como legítimas ou fraudulentas. O objetivo é aplicar técnicas de Machine Learning para identificar padrões, avaliar modelos e entender o impacto de penalidades e limiares em classiﬁcadores.

# Objetivo do Projeto

Explorar o comportamento das transações e a taxa real de fraudes

Criar modelos base e modelos supervisonados

Avaliar desempenho usando métricas adequadas para classes desbalanceadas

Visualizar curvas ROC, recall e análises por limiar

Testar diferentes penalidades e regularizações via grid search

# Sobre o Dataset

O conjunto de dados contém transações bancárias com um rótulo:

```Class = 0``` → transação legítima

```Class = 1``` → fraude

No notebook é calculada a porcentagem real de fraude, que normalmente é extremamente baixa — um desafio central em problemas de detecção de anomalias.

# Tecnologias Utilizadas

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn (para heatmap do Grid Search)

Jupyter Notebook

# Pipeline do Projeto
1. Carregamento e Análise Inicial

Importação do dataset

Visualização inicial

Cálculo da porcentagem de transações fraudulentas

2. Divisão dos Dados
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```
3. Modelo Base – Dummy Classifier

Uso de:

```DummyClassifier(strategy="most_frequent")```

Avaliado com:

Acurácia

Recall

Matriz de confusão

Serve como baseline para comparação.

4. Classificador Real – Regressão Logística

Treinamento com:
```python
LogisticRegression(max_iter=1500, random_state=0)
```

Geração de:

Probabilidades

Curva ROC

Avaliação de trade-off entre FPR e TPR

5. Análise de Penalidades e Limiar

Foi implementada uma função para:

Aplicar diferentes thresholds

Calcular métricas

Gerar uma curva de recall penalizado

Útil para cenários onde falsos negativos têm custo muito alto.

6. Grid Search e Heatmap

Visualização da busca em grade (L1 × L2, diferentes valores de C):

Heatmap usando seaborn

Interpretação visual da melhor combinação de hiperparâmetros

# Resultados Principais

Fraudes representam apenas uma pequena fração dos dados (dataset altamente desbalanceado).

O DummyClassifier alcança alta acurácia, porém recall 0 — mostrando que acurácia não é adequada para esse problema.

A Regressão Logística apresenta desempenho superior, especialmente quando analisada por limiar.

As curvas ROC e recall mostram claramente a sensibilidade do modelo à escolha do threshold.

O Grid Search identifica melhores combinações de regularização para lidar com o desbalanceamento.
