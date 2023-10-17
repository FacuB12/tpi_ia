import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_distribucion_columnas(dataset, nombre_columna):
    # Calcular la media y la desviación estándar de la columna
    mean = np.mean(dataset)
    std_dev = np.std(dataset)

    # Crear un rango de valores para la distribución normal
    x = np.linspace(min(dataset), max(dataset), 1000)

    # Crear un gráfico para la columna
    plt.figure(figsize=(8, 4))
    plt.hist(dataset, bins=20, density=True, alpha=0.6, color='b', label='Histograma de Datos')
    
    # Calcular la función de densidad de probabilidad de la distribución normal
    pdf = norm.pdf(x, mean, std_dev)
    
    # Graficar la distribución normal
    plt.plot(x, pdf, 'r', label='Distribución Normal')
    
    plt.title(f'Distribución Normal de la Columna {nombre_columna}')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    
    plt.show()