import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Caminho para o arquivo de dados
DATA_PATH = "/home/me/dataset/mimic-iii-1.4/reduced30StratNew/CHARTEVENTS.csv"

# Função para carregar dados em chunks e calcular estatísticas básicas
def calculate_stats(chunk, stats_results):
    numeric_columns = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CGID', 'VALUENUM', 'WARNING', 'ERROR']
    chunk = chunk[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Converte todas as colunas para numéricas, valores não numéricos se tornam NaN
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
def process_file(file_path, chunksize=10**5, low_memory=False):
    stats_results = {}
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtraia 1 para ignorar a linha do cabeçalho
    total_chunks = (total_rows // chunksize) + 1

    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize, low_memory=low_memory), total=total_chunks, desc="Processando chunks"):
        stats_results = calculate_stats(chunk, stats_results)
    
    # Calcular a média
    for column in stats_results:
        stats_results[column]["mean"] = stats_results[column]["sum"] / stats_results[column]["count"]
    
    return stats_results

# Processar o arquivo e calcular estatísticas
stats_results = process_file(DATA_PATH, low_memory=False)
print("Estatísticas calculadas:", stats_results)

# Carregar uma amostra para análise exploratória e visualização
sample_df = pd.read_csv(DATA_PATH, nrows=10000, low_memory=False)

# Verificar valores ausentes
missing_data = sample_df.isnull().sum()
print(f"Missing Data:\n{missing_data[missing_data > 0]}")

# Visualização da distribuição das variáveis
sample_df.hist(bins=30, figsize=(15, 10))
plt.show()

# Remover colunas não numéricas antes de calcular a matriz de correlação
numeric_cols = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CGID', 'VALUENUM', 'WARNING', 'ERROR']
sample_df_numeric = sample_df[numeric_cols]

# Matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(sample_df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Verificação de consistência dos dados (exemplo: verificar se idades estão dentro de um intervalo plausível)
if 'age' in sample_df.columns:
    inconsistent_ages = sample_df[(sample_df['age'] < 0) | (sample_df['age'] > 120)]
    print(f"Inconsistent ages:\n{inconsistent_ages}")

# Verificação de outliers
plt.figure(figsize=(12, 8))
sample_df_numeric.boxplot()
plt.title('Boxplot das Variáveis Numéricas')
plt.show()

# Criar boxplots específicos para colunas selecionadas
plt.figure(figsize=(12, 8))
columns_to_plot = ['VALUENUM', 'WARNING', 'ERROR']  # Substitua pelos nomes das colunas que deseja plotar
for col in columns_to_plot:
    if col in sample_df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=sample_df, y=col)
        plt.title(f'Boxplot da coluna: {col}')
        plt.show()
    else:
        print(f"Coluna {col} não encontrada no DataFrame.")
