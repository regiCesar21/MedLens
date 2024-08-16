import matplotlib.pyplot as plt
import numpy as np

# Dados do artigo original
algorithms = ['K-Nearest Neighbors', 'Logistic Regression', 'Random Forest', 'Multilayer Perceptron', 'Gradient Boosting', 'LightGBM', 'XGBoost']
accuracy_original = [0.902, 0.931, 0.953, 0.897, 0.956, 0, 0]

# Seus resultados
accuracy_mine = [0.922, 0.911, 0.938, 0.904, 0.934, 0.932, 0.928]

# Configurações do gráfico
x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))

# Barras de Acurácia
bar1 = ax.bar(x - width/2, accuracy_original, width, label='Acurácia Artigo', color='blue')
bar2 = ax.bar(x + width/2, accuracy_mine, width, label='Acurácia atingida', color='lightblue')

# Adicionar título e rótulos
ax.set_xlabel('Algoritmo')
ax.set_ylabel('Acurácia')
ax.set_title('Comparação de Acurácia dos Algoritmos')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend(loc='lower right')

# Mostrar valores nas barras
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)

# Mostrar gráfico
plt.xticks(rotation=45)
plt.show()
