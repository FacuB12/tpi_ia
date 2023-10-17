import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def entrenar(self, X, y):
        self.X_entrenamiento = X
        self.y_entrenamiento = y

    def distancia_euclidiana(self, x1, x2):
        #if isinstance(x1, (list, np.ndarray)) and len(x1) == 1:
            # x2 = x2.reshape((1, 15))
            # x1 = x1.reshape((1, 15))
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def distancia_manhattan(self, x1, x2):
        # if isinstance(x1, (list, np.ndarray)) and len(x1) == 1:
        #     #  x2 = x2.reshape((1, 15))
        #     #  x1 = x1.reshape((1, 15))
        #     print(x1[0])
        #     print(x2)
        return np.sum(np.abs(x1 - x2))
    
    def hamming_distance(self, x1, x2):
        return np.sum(x1 != x2)



    def predecir(self, X, distance_function):
        if distance_function == 1:
            distance_func = self.distancia_euclidiana
        elif distance_function == 2:
            distance_func = self.distancia_manhattan
        elif distance_function == 3:
            distance_func = self.hamming_distance
        
        if isinstance(X, (list, np.ndarray)) and len(X) > 1:
            # Si X es una lista o un array con más de una fila, iterar sobre las filas
            y_predicho = [self._predecir(x, distance_func) for x in X]
        else:
            y_predicho = [self._predecir(X[0], distance_func)]
        return y_predicho


        # y_predicho = [self._predecir(x, distance_func) for x in X]
        # return np.array(y_predicho)

    def _predecir(self, x, distancia_funcion):
        
        # Calcular las distancias entre x y todos los puntos en el conjunto de entrenamiento
        distancias = [distancia_funcion(x, x_entrenamiento) for x_entrenamiento in self.X_entrenamiento]
        
        # Obtener los índices de los k puntos más cercanos
        k_indices = np.argsort(distancias)[:self.k]
        # Obtener las etiquetas de los k puntos más cercanos
        k_etiquetas_vecinas = [self.y_entrenamiento[i] for i in k_indices]
        # Devolver la etiqueta más común entre los k puntos más cercanos
        mas_comun = np.bincount(k_etiquetas_vecinas).argmax()
        return mas_comun
