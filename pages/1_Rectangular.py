import streamlit as st
import matplotlib.pyplot as plt
from funciones import diagrama_interaccion

st.title("📏 Columna Rectangular")

As_input = st.text_input("Área de acero As (cm², separadas por coma):", "31.68,10.14,10.14,31.68")
As = [float(x.strip()) for x in As_input.split(',') if x.strip()]
b = st.number_input("Ancho (cm):", value=40)
h = st.number_input("Altura (cm):", value=60)
r = st.number_input("Recubrimiento (cm):", value=4.5)
fy = st.number_input("Fy (kg/cm²):", value=4200)
fpc = st.number_input("f'c (kg/cm²):", value=250)
fpc_valido=True
if fpc >= 500 or fpc < 170:
        st.warning("⚠️ f'c fuera del rango válido (170 ≤ f'c < 500)")
        fpc_valido = False
    # Mostrar sección
st.subheader("📊 Sección generada:")
fig, ax = plt.subplots(figsize=(3,3))
ax.plot([0, b, b, 0, 0], [0, 0, h, h, 0], '-o')
ax.set_aspect('equal')
st.pyplot(fig)


if st.button("📈 Generar Diagrama"):
    M, P, c, F_total, d,h = diagrama_interaccion(As, b, h, r, fy, fpc)


    st.session_state["M"] = M
    st.session_state["P"] = P
    st.session_state["c"]=c
    st.session_state["d"]=d
    st.session_state["h"]=h
    st.session_state["F_total"]=F_total
    st.session_state["fy"]=fy
    st.session_state["As"]=As

    st.success("¡Diagrama generado correctamente! Ve a la pestaña 'Resultados'")