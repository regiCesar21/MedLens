import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
        try:
            stats_results[column]["sum"] += chunk[column].sum()
        except TypeError:
            pass  # Ignore columns that cannot be summed
        stats_results[column]["min"] = min(stats_results[column]["min"], chunk[column].min())
        stats_results[column]["max"] = max(stats_results[column]["max"], chunk[column].max())
    return stats_results

# Função para processar o arquivo em chunks
def process_file(file_path, chunksize=10**5, low_memory=False):
    stats_results = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize, low_memory=low_memory), unit="chunk"):
            executor.submit(calculate_stats, chunk, stats_results)
    
    # Calcular a média
    for column in stats_results:
        stats_results[column]["mean"] = stats_results[column]["sum"] / stats_results[column]["count"]
    
    return stats_results

# Função para retornar estatísticas do boxplot
def boxplot_stats(df):
    stats = df.describe()
    return {
        'Min': df.min(),
        '25th percentile (Q1)': stats.loc['25%'],
        'Median (Q2)': stats.loc['50%'],
        '75th percentile (Q3)': stats.loc['75%'],
        'Max': df.max(),
        'Outliers': df[(df < stats.loc['25%']) | (df > stats.loc['75%'])].dropna()
    }

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
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
sample_df_numeric = sample_df[numeric_cols]

# Matriz de correlação (considerando conversão para datetime se CHARTTIME for usado)
if 'CHARTTIME' in sample_df_numeric.columns:
    sample_df_numeric['CHARTTIME'] = pd.to_datetime(sample_df_numeric['CHARTTIME'], errors='coerce').astype('int64')
plt.figure(figsize=(12, 8))
sns.heatmap(sample_df_numeric.corr(), annot=True, cmap='coolwarm')
plt.show()

# Verificar consistência dos dados (exemplo: verificar se idades estão dentro de um intervalo plausível)
if 'age' in sample_df.columns:
    inconsistent_ages = sample_df[(sample_df['age'] < 0) | (sample_df['age'] > 120)]
    print(f"Inconsistent ages:\n{inconsistent_ages}")

# Verificar duplicatas
duplicate_rows = sample_df[sample_df.duplicated()]
print(f"Duplicated rows:\n{duplicate_rows}")

# Análise exploratória de algumas variáveis específicas (exemplo: SpO2)
spo2 = sample_df[sample_df['ITEMID'] == 220277]
plt.figure(figsize=(10, 6))
sns.histplot(spo2['VALUENUM'].dropna(), bins=50, kde=True)
plt.title("Distribuição dos valores de SpO2")
plt.xlabel("SpO2")
plt.ylabel("Frequência")
plt.show()

# Calcular e imprimir estatísticas detalhadas dos boxplots
for col in numeric_cols:
    if col in sample_df.columns:
        stats = boxplot_stats(sample_df[col])
        print(f"\nEstatísticas detalhadas para '{col}':")
        for key, value in stats.items():
            if isinstance(value, pd.Series):  # Se for uma Series, transforma em string
                value_str = ', '.join(map(str, value.tolist()))
                print(f"{key}: {value_str}")
            else:
                print(f"{key}: {value}")

        # Visualização dos outliers
        if not stats['Outliers'].empty:
            print(f"\nOutliers para '{col}':")
            print(stats['Outliers'])
    else:
        print(f"Coluna {col} não encontrada no DataFrame.")
