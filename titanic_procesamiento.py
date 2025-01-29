import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##cargar los datos del titanic
import kagglehub

titanic = pd.read_csv('/content/titanic.csv')

# Título de la aplicación
st.title("Exploración de datos: Titanic")
st.image('/content/titanic.jpg', caption="Titanic")
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

    # Convertir el contenido del buffer a texto
    info_text = buffer.getvalue()
    st.text(info_text)

# Estadísticas descriptivas
if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
    st.write("### Estadísticas descriptivas")
    st.write(titanic.describe())
# Datos faltantes
if st.sidebar.checkbox("Mostrar datos faltantes"):
    st.write("### Datos faltantes por columna")
    st.write(titanic.describe())


# Sección para gráficos dinámicos
st.sidebar.header("Gráficos dinámicos")

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
