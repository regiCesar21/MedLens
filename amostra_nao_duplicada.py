import pandas as pd
import os
from tqdm import tqdm

# Diretório e caminho para o arquivo reduzido
REDUCED_DIR = os.path.expanduser("~/dataset/mimic-iii-1.4/reduced30Strat")
reduced_file_path = os.path.join(REDUCED_DIR, "CHARTEVENTS_reduced.csv")

# Caminho para o arquivo sem duplicatas
nonduplicated_file_path = os.path.join(REDUCED_DIR, "CHARTEVENTS_nonduplicated.csv")

# Função para remover duplicatas no arquivo CSV usando chunks
def remove_duplicates_in_chunks(file_path, output_file_path, chunk_size=10**6):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    partial_file_path = os.path.join(REDUCED_DIR, f"{base_name}_noduplicates_partial.csv")
    
    # Remover arquivo parcial anterior se existir
    if os.path.exists(partial_file_path):
        os.remove(partial_file_path)
    
    # Inicializar a barra de progresso
    pbar = tqdm(total=os.path.getsize(file_path), unit='B', unit_scale=True, desc='Processando')
    
    # Inicializar contadores
    initial_rows = 0
    final_rows = 0
    
    # Iterar sobre o arquivo CSV em chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Verificar a quantidade de linhas inicialmente
        initial_rows += len(chunk)

        # Remover duplicatas no chunk
        chunk.drop_duplicates(inplace=True)
        
        # Verificar a quantidade de linhas após remoção de duplicatas
        final_rows += len(chunk)
        
        # Salvar o chunk sem duplicatas para o arquivo parcial
        chunk.to_csv(partial_file_path, mode='a', header=not initial_rows, index=False)
        
        # Atualizar a barra de progresso
        pbar.update(chunk_size)
    
    # Fechar a barra de progresso
    pbar.close()

    # Informações sobre a redução
    print(f"Número inicial de linhas: {initial_rows}")
    print(f"Número final de linhas após remoção de duplicatas: {final_rows}")
    print(f"Linhas removidas: {initial_rows - final_rows}")

    # Renomear o arquivo parcial para o arquivo sem duplicatas
    os.rename(partial_file_path, output_file_path)

# Executar a remoção de duplicatas e salvar no arquivo CHARTEVENTS_nonduplicated.csv
remove_duplicates_in_chunks(reduced_file_path, nonduplicated_file_path)
