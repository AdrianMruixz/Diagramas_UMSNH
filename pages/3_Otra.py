import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funciones import diagrama_interaccion_cualquiera



st.title("Otra forma")
st.subheader("Datos de entrada para columna personalizada")
As_input = st.text_input("Área de acero As (separadas por coma):", "31.68, 10.14, 10.14, 31.68")
As = np.array([float(i.strip()) for i in As_input.split(',') if i.strip() != ''])
h = st.number_input("Altura máxima (cm):", value=60)
r = st.number_input("Recubrimiento (cm):", value=4.5)
fy = st.number_input("Fy (kg/cm²):", value=4200)
fpc = st.number_input("F'c (kg/cm²):", value=250)
fpc_valido=True
if fpc >= 500 or fpc < 170:
        st.warning("⚠️ f'c fuera del rango válido (170 ≤ f'c < 500)")
        fpc_valido = False

st.subheader("Coordenadas de vértices (antihorario)")
verts_df = st.data_editor(pd.DataFrame({'x': [0, 30, 30, 0], 'y': [0, 0, 60, 60]}), num_rows="dynamic")
verts = verts_df[['x', 'y']].values.tolist()

fig, ax = plt.subplots(figsize=(3,3))
np_verts = np.array(verts + [verts[0]])
ax.plot(np_verts[:, 0], np_verts[:, 1], '-o')
ax.set_aspect('equal')
st.pyplot(fig)
generar=False
if st.button("Generar Diagrama"):
        generar = True
        if fpc_valido:
            M, P, c, F_total, d,h = diagrama_interaccion_cualquiera(As, verts, h, r, fy, fpc)

# --- SI SE GENERÓ DIAGRAMA ---
if generar and fpc_valido:
    st.session_state["M"] = M
    st.session_state["P"] = P
    st.session_state["c"] = c
    st.session_state["d"]=d
    st.session_state["h"]=h
    st.session_state["F_total"]=F_total
    st.session_state["fy"]=fy
    st.session_state["As"]=As
    st.success("¡Diagrama generado correctamente! Ve a la pestaña 'Resultados'")