import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

##cargar los datos del titanic
import kagglehub

titanic = pd.read_csv('titanic.csv')

# Título de la aplicación
st.title("Exploración de datos: Titanic")
st.image('titanic.jpg', caption="Titanic")
# Descripción inicial

st.write("""
### ¡Bienvenidos!
Esta aplicación interactiva permite explorar el dataset de Titanic.
Puedes:
1. Ver los primeros registros.
2. Consultar información general del dataset.
3. Generar gráficos dinámicos.
""")

# Sección para explorar el dataset
st.sidebar.header("Exploración de datos")

# Mostrar las primeras filas dinámicamente
if st.sidebar.checkbox("Mostrar primeras filas"):
    n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, 50, 5)
    st.write(f"### Primeras {n_rows} filas del dataset")
    st.write(titanic.head(n_rows))


# Mostrar información del dataset
import io
if st.sidebar.checkbox("Mostrar información del dataset"):
    st.write("### Información del dataset")

    # Capturar la salida de info() en un buffer
    buffer = io.StringIO()
    titanic.info(buf=buffer)
    
    # Procesar la salida para estructurarla mejor
    info_text = buffer.getvalue().split("\n")  # Dividir en líneas
    info_text = [line.strip() for line in info_text if line.strip()]  # Quitar espacios vacíos
    
    # Extraer información clave
    filas_columnas = info_text[0]  # Primera línea con shape
    columnas_info = info_text[3:]  # A partir de la cuarta línea están las columnas

    # Mostrar filas y columnas
    st.write(f"**{filas_columnas}**")

    # Convertir la información de columnas en un DataFrame
    column_data = []
    for line in columnas_info:
        parts = line.split()  # Separar por espacios
        if len(parts) >= 3:
            column_name = parts[1]  # Nombre de la columna
            non_null_count = parts[2]  # Cantidad de valores no nulos
            dtype = parts[-1]  # Tipo de dato
            column_data.append([column_name, non_null_count, dtype])

    df_info = pd.DataFrame(column_data, columns=["Columna", "No Nulos", "Tipo de Dato"]).iloc[2:]
    memory_values = df_info.iloc[-1].values
    memorie_use = " ".join(str(value) for value in memory_values)
    # Mostrar la tabla en Streamlit
    st.dataframe(df_info.iloc[:-2])
    st.write(f"Uso en memoria {memorie_use}")

# Estadísticas descriptivas
if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
    st.write("### Estadísticas descriptivas")
    st.write(titanic.describe())
    
# Datos faltantes
if st.sidebar.checkbox("Mostrar datos faltantes"):
    st.write("### Datos faltantes por columna")
    selected_column = st.selectbox("Selecciona una columna para ver los datos faltantes:", titanic.columns)

    # Calcular datos faltantes
    missing_values = titanic[selected_column].isnull().sum()
    total_values = len(titanic[selected_column])
    missing_percentage = (missing_values / total_values) * 100

    # Mostrar resultado
    st.write(f"### Información de la columna: `{selected_column}`")
    st.write(f"- **Valores totales:** {total_values}")
    st.write(f"- **Valores faltantes:** {missing_values} ({missing_percentage:.2f}%)")
    
    if st.button("Mostrar todos los valores faltantes"):
        missing_total = titanic.isnull().sum()
        missing_total_df = pd.DataFrame({"Columna": missing_total.index, "Valores Faltantes": missing_total.values})
        
        # Filtrar solo las columnas con valores faltantes
        missing_total_df = missing_total_df[missing_total_df["Valores Faltantes"] > 0]
        st.write(missing_total_df)

#Frecuencia Columnas
if st.sidebar.checkbox("Frecuencia columnas"):
    st.write("### Frecuencia por columna")
    columna_seleccionada = st.selectbox("Selecciona una columna para ver su frecuencia:", titanic.columns.iloc[:, 1:])
    st.write(titanic[columna_seleccionada].value_counts())
    if st.button("Mostrar valor más frecuente"):
        st.write(titanic[columna_seleccionada].mode()[0])

#Informacion por pasajero
if st.sidebar.checkbox("Información pasajeros"):
    st.write("### Informacion por pasajero")
    row_index = st.number_input("Ingresa el índice de la fila a visualizar:", min_value=0, max_value=len(titanic)-1, step=1)

    if st.button("Mostrar fila seleccionada"):
        st.write(f"### Datos de la fila `{row_index}`")
        st.dataframe(titanic.iloc[[row_index]].iloc[:, 1:])

#Matriz de correlacion
if st.sidebar.checkbox("Matriz de correlacion"):
    st.write("### Matriz de correlacion")
    # Filtrar solo las columnas numéricas
    titanic_num = titanic.select_dtypes(include=['float64', 'int64'])
    
    # Calcular la matriz de correlación
    correlacion = titanic_num.corr()
    
    # Configuración de la gráfica
    plt.figure(figsize=(10, 8))  # Tamaño de la figura
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    
    # Título de la gráfica
    plt.title('Matriz de Correlación de Titanic')
    
    # Mostrar la gráfica en Streamlit
    st.pyplot(plt)


# Copiar el DataFrame para evitar modificar el original
titanic_copy = titanic.copy()

# Estrategias disponibles
estrategias = ['Media', 'Mediana', 'Moda', 'Eliminar filas', 'Eliminar columna', 'KNN Imputación']
if st.sidebar.checkbox("Imputacion de datos"):
    # Crear selectboxes para seleccionar estrategias
    fare_strategy = st.selectbox('Estrategia para Fare:', estrategias, index=estrategias.index('Media'))
    age_strategy = st.selectbox('Estrategia para Age:', estrategias, index=estrategias.index('Media'))
    cabin_strategy = st.selectbox('Estrategia para Cabin:', estrategias, index=estrategias.index('Media'))
    
    # Botón para aplicar las estrategias
    if st.button('Aplicar estrategias'):
        df_2 = titanic_copy.copy()  # Trabajar sobre una copia del DataFrame
    
        # Imputador KNN (se aplicará solo si es seleccionado)
        knn_imputer = KNNImputer(n_neighbors=5)
    
        # Aplicar estrategia para 'Fare'
        if fare_strategy == 'Media':
            df_2['Fare'].fillna(df_2['Fare'].mean(), inplace=True)
        elif fare_strategy == 'Mediana':
            df_2['Fare'].fillna(df_2['Fare'].median(), inplace=True)
        elif fare_strategy == 'Moda':
            df_2['Fare'].fillna(df_2['Fare'].mode()[0], inplace=True)
        elif fare_strategy == 'Eliminar filas':
            df_2.dropna(subset=['Fare'], inplace=True)
        elif fare_strategy == 'Eliminar columna':
            df_2.drop(columns=['Fare'], inplace=True)
        elif fare_strategy == 'KNN Imputación':
            df_2[['Fare']] = knn_imputer.fit_transform(df_2[['Fare']])
    
        # Aplicar estrategia para 'Age'
        if age_strategy == 'Media':
            df_2['Age'].fillna(df_2['Age'].mean(), inplace=True)
        elif age_strategy == 'Mediana':
            df_2['Age'].fillna(df_2['Age'].median(), inplace=True)
        elif age_strategy == 'Moda':
            df_2['Age'].fillna(df_2['Age'].mode()[0], inplace=True)
        elif age_strategy == 'Eliminar filas':
            df_2.dropna(subset=['Age'], inplace=True)
        elif age_strategy == 'Eliminar columna':
            df_2.drop(columns=['Age'], inplace=True)
        elif age_strategy == 'KNN Imputación':
            df_2[['Age']] = knn_imputer.fit_transform(df_2[['Age']])
    
        # Aplicar estrategia para 'Cabin'
        if cabin_strategy == 'Media' or cabin_strategy == 'Mediana':
            st.write("❌ La estrategia seleccionada no es válida para 'Cabin' porque la variable no es numérica.")
        else:
            if cabin_strategy == 'Moda':
                df_2['Cabin'].fillna(df_2['Cabin'].mode()[0], inplace=True)
            elif cabin_strategy == 'Eliminar filas':
                df_2.dropna(subset=['Cabin'], inplace=True)
            elif cabin_strategy == 'Eliminar columna':
                df_2.drop(columns=['Cabin'], inplace=True)
            elif cabin_strategy == 'KNN Imputación':
                # Cabin tiene muchos valores únicos, simplificaremos usando un reemplazo categórico
                df_2['Cabin'] = df_2['Cabin'].str[0]  # Usar solo la primera letra de Cabin
                df_2['Cabin'] = df_2['Cabin'].fillna('U')  # Rellenar NaN con una categoría ficticia
    
        # Mostrar el DataFrame resultante
        st.write("### DataFrame resultante")
        st.dataframe(df_2.head())

        st.markdown("""
        ## **Retroalimentación: ¿qué era mejor escoger?**
        
        ### 1. Manejo de valores faltantes:
        
        - **Age** (faltan 86 valores, 20.57%):
          - Se recomienda **imputación por KNN**, ya que considera relaciones entre columnas como Pclass, Sex, y SibSp para estimar las edades de manera más precisa. Esta estrategia es útil porque la edad es una característica importante para predecir la supervivencia.
          
        - **Fare** (falta 1 valor, 0.24%):
          - Usar imputación por **media o mediana**, ya que el número de valores faltantes es insignificante. También una buena alternativa es **eliminar la fila** con el valor faltante.
        
        - **Cabin** (faltan 327 valores, 78.23%):
          - Dado que el porcentaje de valores faltantes es extremadamente alto, se recomienda **eliminar la columna**. Aunque los valores de Cabin podrían contener información útil, el esfuerzo necesario para imputar o agrupar los datos supera los beneficios para la mayoría de los análisis iniciales.
        
        ### 2. Codificación de variables categóricas: 
        Usar **OneHot Encoding**, ya que hay pocas columnas categóricas con un número bajo de clases únicas. Esto evita que el modelo interprete relaciones ordinales inexistentes entre las categorías.
        
        ### 3. Estandarización de datos: 
        Usar **StandardScaler**, que ajusta los datos para tener **media 0** y **desviación estándar 1**. Esto es particularmente útil para modelos como **regresión logística**, **SVM**, o **redes neuronales**, donde la escala uniforme mejora la convergencia y el desempeño del modelo.
        """)
        







# Sección para gráficos dinámicos
if st.sidebar.checkbox("Gráficos dinámicos"):

    # Selección de variables para el gráfico
    x_var = st.sidebar.selectbox("Selecciona la variable X:", titanic.columns)
    y_var = st.sidebar.selectbox("Selecciona la variable Y:", titanic.columns)
    
    # Tipo de gráfico
    chart_type = st.sidebar.radio(
        "Selecciona el tipo de gráfico:",
        ("Dispersión", "Histograma", "Boxplot")
    )
    
    # Mostrar el gráfico
    st.write("### Gráficos")
    if chart_type == "Dispersión":
        st.write(f"#### Gráfico de dispersión: {x_var} vs {y_var}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=titanic, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Histograma":
        st.write(f"#### Histograma de {x_var}")
        fig, ax = plt.subplots()
        sns.histplot(titanic[x_var], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot":
        st.write(f"#### Boxplot de {y_var} por {x_var}")
        fig, ax = plt.subplots()
        sns.boxplot(data=titanic, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)
