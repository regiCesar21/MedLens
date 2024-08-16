import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Diretórios locais
DATA_DIR = "~/dataset/mimic-iii-1.4/"
REDUCED_DIR = os.path.join(DATA_DIR, "reduced30Strat")  # Diretório para os arquivos reduzidos

# Criar diretório "reduced30Strat" se não existir
os.makedirs(REDUCED_DIR, exist_ok=True)

# Caminhos para os arquivos
chartevents_path = os.path.join(DATA_DIR, 'CHARTEVENTS.csv')
# labevents_path = os.path.join(DATA_DIR, 'LABEVENTS.csv')
# admissions_path = os.path.join(DATA_DIR, 'ADMISSIONS.csv')

# Função para selecionar uma amostra estratificada em chunks
def stratified_sample_in_chunks(file_path, sample_size, stratify_col, chunk_size=10**5):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    partial_file_path = os.path.join(REDUCED_DIR, f"{base_name}_reduced.csv")
    
    # Remover arquivo parcial anterior se existir
    if os.path.exists(partial_file_path):
        os.remove(partial_file_path)
    
    # Lista para guardar os chunks amostrados
    chunk_list = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        stratify_col_values = chunk[stratify_col]
        
        # Verificar se há mais de um valor único para estratificação
        if stratify_col_values.nunique() > 1:
            try:
                # Amostragem estratificada usando train_test_split
                _, sample_chunk = train_test_split(chunk, test_size=sample_size, stratify=stratify_col_values)
            except ValueError:
                # Se a estratificação falhar (caso raro), amostrar aleatoriamente
                sample_chunk = chunk.sample(frac=sample_size)
        else:
            # Amostrar aleatoriamente se houver apenas um valor único
            sample_chunk = chunk.sample(frac=sample_size)
        
        # Adicionar o chunk amostrado à lista
        chunk_list.append(sample_chunk)
    
    # Concatenar todos os chunks amostrados em um único DataFrame
    sample_df = pd.concat(chunk_list, axis=0)
    
    # Salvar o arquivo reduzido final
    sample_df.to_csv(partial_file_path, index=False)
    
    return partial_file_path

# Proporção de amostra desejada (30%)
sample_size = 0.3

# Selecionar uma amostra estratificada de 30% dos dados para cada arquivo
reduced_chartevents_path = stratified_sample_in_chunks(chartevents_path, sample_size, stratify_col='ITEMID')
# reduced_labevents_path = stratified_sample_in_chunks(labevents_path, sample_size, stratify_col='ITEMID')

# Copiar o arquivo ADMISSIONS.csv para o diretório reduzido
# admissions_reduced_path = os.path.join(REDUCED_DIR, 'ADMISSIONS.csv')
os.system(f'cp {admissions_path} {admissions_reduced_path}')

# Imprimir os caminhos dos arquivos reduzidos
print(f"Amostra estratificada de 30% de CHARTEVENTS.csv salva em: {reduced_chartevents_path}")
# print(f"Amostra estratificada de 30% de LABEVENTS.csv salva em: {reduced_labevents_path}")
# print(f"Arquivo ADMISSIONS.csv copiado para: {admissions_reduced_path}")
