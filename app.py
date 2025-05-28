import streamlit as st

st.set_page_config(page_title="App de Diagramas", layout="wide")

st.title("📐 App Profesional de Diagramas de Interacción")
st.markdown("""
Bienvenido a la **App de Columnas de Concreto Reforzado**  
Usa el menú lateral para seleccionar el tipo de sección.  
Esta app calcula y genera diagramas de interacción para columnas:
- Rectangulares
- Circulares
- Personalizadas

Desarrollado por: *Adrian Magaña Ruiz*
""")
# Menú lateral real
opcion = st.sidebar.radio("Selecciona el tipo de columna", ["Rectangular", "Circular", "Otra"])



st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Concrete_columns_Columbia_Square.jpg/640px-Concrete_columns_Columbia_Square.jpg", width=700)