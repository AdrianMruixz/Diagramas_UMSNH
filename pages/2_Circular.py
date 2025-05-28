import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones import diagrama_interaccion_circular
st.title( "Circular")
st.subheader("Datos de entrada para columna circular")
As_input = st.text_input("Área de acero As (separadas por coma):", "31.68, 10.14, 10.14, 31.68")
As = np.array([float(i.strip()) for i in As_input.split(',') if i.strip() != ''])
D = st.number_input("Diámetro de la sección (cm):", value=60)
r = st.number_input("Recubrimiento (cm):", value=4.5)
fy = st.number_input("Fy (kg/cm²):", value=4200)
fpc = st.number_input("F'c (kg/cm²):", value=250)
n = st.slider("Número de puntos para aproximación", 8, 100, 60)
fpc_valido=True
if fpc >= 500 or fpc < 170:
        st.warning("⚠️ f'c fuera del rango válido (170 ≤ f'c < 500)")
        fpc_valido = False
h=D
d = D - r
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
x = (d / 2) * np.cos(theta) + d / 2
y = (d / 2) * np.sin(theta) + d / 2
verts = np.column_stack((x, y))

st.subheader("📊 Sección generada:")
fig, ax = plt.subplots(figsize=(3,3))
ax.plot(np.append(x, x[0]), np.append(y, y[0]), '-o')
ax.set_aspect('equal')
st.pyplot(fig)
if st.button("Generar Diagrama"):
        generar = True
        if fpc_valido:
            M, P, c, F_total, d,h= diagrama_interaccion_circular(As, D, r, fy, fpc)

        st.session_state["d"]=d
        st.session_state["h"]=h
        st.session_state["M"] = M
        st.session_state["P"] = P
        st.session_state["c"]=c
        st.session_state["F_total"]=F_total
        st.session_state["fy"]=fy
        st.session_state["As"]=As
        st.success("¡Diagrama generado correctamente! Ve a la pestaña 'Resultados'")
