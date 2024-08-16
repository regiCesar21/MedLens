import dask.dataframe as dd
from dask.distributed import Client
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Configurar o cliente Dask
client = Client()

# Caminho para o arquivo de dados
DATA_PATH = "~/dataset/mimic-iii-1.4/reduced30StratNew/CHARTEVENTS.csv"

# Carregar o DataFrame com Dask
ddf = dd.read_csv(DATA_PATH, assume_missing=True)

# Selecionar apenas colunas numéricas
numeric_cols = ddf.select_dtypes(include=[float, int]).columns
ddf_numeric = ddf[numeric_cols]

# Converter para um DataFrame pandas para processamento
df_numeric = ddf_numeric.compute()

# Normalizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Aplicar o PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Criar um DataFrame com os componentes principais
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Visualizar os resultados do PCA
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2 Componentes Principais')
plt.show()

# Explicar a variância
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
