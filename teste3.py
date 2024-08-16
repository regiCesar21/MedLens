import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Caminho para os arquivos de dados
CHARTEVENTS_PATH = "/home/me/dataset/mimic-iii-1.4/reduced30StratNew/CHARTEVENTS.csv"
LABEVENTS_PATH = "/home/me/dataset/mimic-iii-1.4/reduced30StratNew/LABEVENTS.csv"

# Função para carregar dados em chunks e calcular estatísticas básicas
def calculate_stats(chunk, stats_results):
    numeric_columns = [col for col in chunk.columns if pd.api.types.is_numeric_dtype(chunk[col])]
    chunk = chunk[numeric_columns]
    chunk = chunk.apply(pd.to_numeric, errors='coerce')  # Converte todas as colunas para numéricas, valores não numéricos se tornam NaN
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

# Processar o arquivo CHARTEVENTS e calcular estatísticas
chartevents_stats_results = process_file(CHARTEVENTS_PATH, low_memory=False)
print("Estatísticas CHARTEVENTS calculadas:", chartevents_stats_results)

# Carregar uma amostra para análise exploratória e visualização
chartevents_sample_df = pd.read_csv(CHARTEVENTS_PATH, nrows=10000, low_memory=False)

# Processar o arquivo LABEVENTS e calcular estatísticas
labevents_stats_results = process_file(LABEVENTS_PATH, low_memory=False)
print("Estatísticas LABEVENTS calculadas:", labevents_stats_results)

# Carregar uma amostra para análise exploratória e visualização
labevents_sample_df = pd.read_csv(LABEVENTS_PATH, nrows=10000, low_memory=False)

# Fazer a junção das amostras com base em SUBJECT_ID
combined_df = pd.merge(chartevents_sample_df, labevents_sample_df, on='SUBJECT_ID', suffixes=('_chartevents', '_labevents'))

# Verificar valores ausentes
missing_data = combined_df.isnull().sum()
print(f"Missing Data:\n{missing_data[missing_data > 0]}")

# Visualização da distribuição das variáveis
combined_df.hist(bins=30, figsize=(15, 10))
plt.show()

# Remover colunas não numéricas antes de calcular a matriz de correlação
numeric_cols = ['ROW_ID_chartevents', 'SUBJECT_ID', 'HADM_ID_chartevents', 'ICUSTAY_ID', 'ITEMID_chartevents', 'CGID', 'VALUENUM_chartevents', 'VALUENUM_labevents', 'WARNING', 'ERROR']
combined_df_numeric = combined_df[numeric_cols]

# Matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(combined_df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Verificação de consistência dos dados (exemplo: verificar se idades estão dentro de um intervalo plausível)
if 'age_chartevents' in combined_df.columns:
    inconsistent_ages = combined_df[(combined_df['age_chartevents'] < 0) | (combined_df['age_chartevents'] > 120)]
    print(f"Inconsistent ages:\n{inconsistent_ages}")

# Verificação de outliers e boxplot mais estilizado para VALUENUM_chartevents
plt.figure(figsize=(12, 8))
sns.boxplot(data=combined_df, y='VALUENUM_chartevents')
plt.title('Boxplot de VALUENUM (CHARTEVENTS)')
plt.xlabel('Distribuição de VALUENUM (CHARTEVENTS)')
plt.ylabel('VALUENUM_chartevents')
sns.despine(trim=True)  # Remove as bordas superiores e direitas
plt.grid(True, linestyle='--', alpha=0.7)  # Adiciona uma grade leve
plt.show()

# Função para imprimir estatísticas detalhadas dos boxplots
def print_boxplot_stats(df, col_name):
    stats = boxplot_stats(df[col_name])
    print(f"\nEstatísticas detalhadas para '{col_name}':")
    for key, value in stats.items():
        if isinstance(value, pd.Series):  # Se for uma Series, transforma em string
            value_str = ', '.join(map(str, value.tolist()))
            print(f"{key}: {value_str}")
        else:
            print(f"{key}: {value}")

    # Visualização dos outliers
    if not stats['Outliers'].empty:
        print(f"\nOutliers para '{col_name}':")
        print(stats['Outliers'])

# Exemplo de impressão de estatísticas detalhadas para VALUENUM_chartevents
print_boxplot_stats(combined_df, 'VALUENUM_chartevents')
