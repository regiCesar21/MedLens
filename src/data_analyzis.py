import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Caminho para o arquivo de dados
DATA_PATH = "~/dataset/mimic-iii-1.4/reduced30StratNew/CHARTEVENTS.csv"

# Função para carregar dados em chunks e calcular estatísticas básicas
def calculate_stats(chunk, stats_results):
    for column in chunk.columns:
        if column not in stats_results:
            stats_results[column] = {
                "count": 0, "sum": 0, "min": np.inf, "max": -np.inf, "mean": 0
            }
        stats_results[column]["count"] += chunk[column].count()
        stats_results[column]["sum"] += chunk[column].sum()
        stats_results[column]["min"] = min(stats_results[column]["min"], chunk[column].min())
        stats_results[column]["max"] = max(stats_results[column]["max"], chunk[column].max())
    return stats_results

# Função para processar o arquivo em chunks
def process_file(file_path, chunksize=10**5):
    stats_results = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize), unit="chunk"):
            executor.submit(calculate_stats, chunk, stats_results)
    
    # Calcular a média
    for column in stats_results:
        stats_results[column]["mean"] = stats_results[column]["sum"] / stats_results[column]["count"]
    
    return stats_results

# Processar o arquivo e calcular estatísticas
stats_results = process_file(DATA_PATH)
print("Estatísticas calculadas:", stats_results)

# Carregar uma amostra para análise exploratória e visualização
sample_df = pd.read_csv(DATA_PATH, nrows=10000)

# Verificar valores ausentes
missing_data = sample_df.isnull().sum()
print(f"Missing Data:\n{missing_data[missing_data > 0]}")

# Visualização da distribuição das variáveis
sample_df.hist(bins=30, figsize=(15, 10))
plt.show()

# Matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(sample_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Verificar consistência dos dados (exemplo: verificar se idades estão dentro de um intervalo plausível)
inconsistent_ages = sample_df[(sample_df['age'] < 0) | (sample_df['age'] > 120)]
print(f"Inconsistent ages:\n{inconsistent_ages}")

# Verificar duplicatas
duplicate_rows = sample_df[sample_df.duplicated()]
print(f"Duplicated rows:\n{duplicate_rows}")

# Dividir os dados em conjuntos de treinamento e teste
X = sample_df.drop('target', axis=1)  # Substitua 'target' pelo nome da coluna alvo
y = sample_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo de exemplo (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred, average="weighted")}')
print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
print(f'Average Precision: {average_precision_score(y_test, y_pred)}')

# Avaliação com validação cruzada
cross_val_scores = cross_val_score(model, X, y, cv=5)
print(f'Validação cruzada scores: {cross_val_scores}')
print(f'Média da validação cruzada: {cross_val_scores.mean()}')

# Processamento e treinamento em chunks para grandes datasets
chunk_size = 10**5  # Ajuste conforme necessário
model = RandomForestClassifier(random_state=42)

for chunk in tqdm(pd.read_csv(DATA_PATH, chunksize=chunk_size), unit='chunk'):
    X_chunk = chunk.drop('target', axis=1)  # Substitua 'target' pelo nome da coluna alvo
    y_chunk = chunk['target']
    model.partial_fit(X_chunk, y_chunk, classes=np.unique(y))

# Avaliação final
y_pred = model.predict(X_test)
print(f'Final Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Final F1 Score: {f1_score(y_test, y_pred, average="weighted")}')
print(f'Final ROC AUC: {roc_auc_score(y_test, y_pred)}')
print(f'Final Average Precision: {average_precision_score(y_test, y_pred)}')