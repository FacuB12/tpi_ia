import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from KNN2 import KNN


st.set_page_config(
    page_title="App de Predicción con K-Nearest Neighbors",
    page_icon="../img/favicon.ico",  
)

st.title("Predicción de Riesgo Cardiovascular")
st.sidebar.image("../img/logo.png")
st.sidebar.header("Descripcion del Dataset")
#st.sidebar.subheader("Cargar Datos")

df = pd.read_csv('../datasets/DataFinal.csv') #Cargar el dataset de entrenamiento


if df is not None:
    st.sidebar.write("Información del dataset:")
    st.sidebar.write("Número de filas:", df.shape[0])
    st.sidebar.write("Número de columnas:", df.shape[1])

    columna_objetivo = df.columns[-1]

    X = df.drop(df.columns[-1], axis=1)
    X = X.drop(df.columns[0], axis=1)
    y = df[columna_objetivo]


    variable=df.to_numpy()

    divisor = int(variable.shape[0] * 0.70)

    matriz_clases = variable[:, -1]
    matriz_atributos = variable [:, 1:-1]

    matriz_clases_entrenamiento = matriz_clases[1 : divisor]
    matriz_clases_prueba = matriz_clases[divisor : ]

    matriz_atributos_entrenamiento = matriz_atributos [1 : divisor]
    matriz_atributos_test = matriz_atributos[ divisor : ]
    
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("Valores del algoritmo implementado:")
    st.sidebar.write("Valor de k:", 9)
    st.sidebar.write("Distancia utilizada: Manhattan")
    knn = KNN(k=9)
    knn.entrenar(matriz_atributos_entrenamiento, matriz_clases_entrenamiento)
    Etiquetas_Predichas = knn.predecir(matriz_atributos_test,2)
    

    precision = accuracy_score(matriz_clases_prueba, Etiquetas_Predichas)
    st.sidebar.write("Precisión del modelo:", precision)

    st.header("Ingresar un nuevo registro para predecir")
    st.text("Consideraciones: \n" +
            "   - Para sexo Femenino ingrese 0 \n" +
            "   - Para sexo Masculino ingrese 1 \n" +
            "   - Para fumador ingrese 1 \n" +
            "   - Para no fumador ingrese 0 \n")

    new_data = {}
    for column in X.columns:
        new_data[column] = st.number_input(f"Ingrese el valor para {column}", value=0.0, step=0.1)

    if st.button("Predecir"):
        new_df = pd.DataFrame([new_data])
        data = new_df.to_numpy()
        prediction = knn.predecir(data,2)
        #st.write(f"El registro ingresado es: {data[0]}")
        if prediction[0] == 0:
            st.write("Prediccion: El paciente no tiene riesgo cardiovascular")
        else:
            st.write("Prediccion: El paciente tiene riesgo cardiovascular")
        



  