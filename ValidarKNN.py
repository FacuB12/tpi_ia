import numpy as np
from KNN2 import KNN
from sklearn.model_selection import KFold



def validar_knn(matriz_atributos_entrenamiento, matriz_clases_entrenamiento, distance_function):
    # Inicializa una lista para almacenar los puntajes promedio para cada valor de k
    puntajes_promedio = []
    valores_k = list(range(1, 101))
    num_divisiones=5
    # Realiza la validación cruzada para cada valor de k
    for kk in valores_k:
        puntajes_fold = []  # Almacena los puntajes de cada fold

        # Configura la validación cruzada K-Fold
        kf = KFold(n_splits=num_divisiones, shuffle=True, random_state=42)  # Puedes ajustar el número de divisiones aquí

        for train_index, test_index in kf.split(matriz_atributos_entrenamiento):
            X_entrenamiento, X_prueba = matriz_atributos_entrenamiento[train_index], matriz_atributos_entrenamiento[test_index]
            y_entrenamiento, y_prueba = matriz_clases_entrenamiento[train_index], matriz_clases_entrenamiento[test_index]

        # Crea y entrena tu modelo KNN con el valor actual de k
        modelo_knn = KNN(k=kk)
        modelo_knn.entrenar(X_entrenamiento, y_entrenamiento)
        
        # Llama a la función predecir con la función de distancia deseada (por ejemplo, "euclidean")
        y_predicho = modelo_knn.predecir(X_prueba, distance_function)
        precision = np.mean(y_predicho == y_prueba)
        
        puntajes_fold.append(precision)

        # Calcula el puntaje promedio para este valor de k
        puntaje_promedio = np.mean(puntajes_fold)
        puntajes_promedio.append(puntaje_promedio)

    # Encuentra el valor de k que produce el mejor puntaje promedio
    mejor_k = valores_k[np.argmax(puntajes_promedio)]
    mejor_puntaje = max(puntajes_promedio)

    resultados = {
        "valores_k": valores_k,
        "puntajes_promedio": puntajes_promedio,
        "mejor_k": mejor_k,
        "mejor_puntaje": mejor_puntaje
    }

    return resultados
