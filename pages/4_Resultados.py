import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from funciones import ajustar_modeloM, ajustar_modeloP,predecir_c
import numpy as np

st.title("📊 Resultados y Diagrama de Interacción")

if "M" in st.session_state and "P" in st.session_state and "c" in st.session_state:
    M = st.session_state["M"]
    P = st.session_state["P"]
    c = st.session_state["c"]
    d=st.session_state["d"]
    h=st.session_state["h"]
    F_total=st.session_state["F_total"]
    fy=st.session_state["fy"]
    As=st.session_state["As"]

    # Crear figura
    fig, ax = plt.subplots()
    ax.plot(M, P, label="Diagrama de Interacción", color='blue')
    ax.set_xlabel("Momento (kg-cm)")
    ax.set_ylabel("Carga axial (kg)")
    ax.set_title("Diagrama de Interacción P-M")
    ax.grid(True)

    # Inputs para agregar puntos adicionales
    st.subheader("➕ Agregar puntos al diagrama")
    with st.expander("🔴 Ingresar puntos manualmente"):
        num_puntos = st.number_input("Número de puntos a agregar", min_value=1, max_value=10, value=1, step=1)
        puntos_M = []
        puntos_P = []
        for i in range(num_puntos):
            col1, col2 = st.columns(2)
            with col1:
                m_val = st.number_input(f"Momento M{i+1} (kg-cm)", key=f"m{i}")
            with col2:
                p_val = st.number_input(f"Carga P{i+1} (kg)", key=f"p{i}")
            puntos_M.append(m_val)
            puntos_P.append(p_val)

    # Gráfica actualizada
    fig, ax = plt.subplots()
    ax.plot(M, P, label="Diagrama de Interacción", color='blue')
    ax.set_xlim(min(M), max(M))
    ax.set_ylim(min(P), max(P))

    if puntos_M and puntos_P:
        ax.scatter(puntos_M, puntos_P, color='red', label='Puntos evaluados', zorder=5)

    ax.set_xlabel("Momento (kg-cm)")
    ax.set_ylabel("Carga axial (kg)")
    ax.set_title("Diagrama de Interacción P-M")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # 🔄 Calcular M o P
    st.subheader("📐 Calcular M o P con modelos ajustados")
    opcion = st.radio("¿Qué deseas calcular?", ("Dado P → Calcular M", "Dado M → Calcular P"))

    if opcion == "Dado P → Calcular M":
        entrada_p = st.number_input("📌 Ingresa P (kg):", value=100.0)
        modeloM, r2_M, _ = ajustar_modeloM(P, M)
        resultado_m = modeloM(entrada_p)
        st.success(f"🔍 Momento estimado: {resultado_m:.2f} kg-cm")
        st.caption(f"Coeficiente de determinación R² del modelo: {r2_M:.4f}")

    elif opcion == "Dado M → Calcular P":
        entrada_m = st.number_input("📌 Ingresa M (kg-cm):", value=100.0)
        modeloP, r2_P, _ = ajustar_modeloP(P, M)
        resultado_p = modeloP(entrada_m)
        st.success(f"🔍 Carga axial estimada: {resultado_p:.2f} kg")
        st.caption(f"Coeficiente de determinación R² del modelo: {r2_P:.4f}")
    #Puntos importantes
    st.subheader("Salida de puntos importantes del diagrama")
    modelo, r2, coef = ajustar_modeloM(P, M)
    raices = np.roots(coef)
    raices_real = raices[np.isreal(raices)].real
    Tabla = np.array([M, P, c])

    puntos = {
            "Descripción": ["Compresión Pura", "Tension Pura", "Flexión Pura", " Comportamiento Balanceado"],
            "Momento (kg-cm)": [0, 0, raices_real, Tabla[0, np.argmin(np.abs(c - 0.6 * h))]],
            "Carga Axial (kg)": [F_total, np.sum(As) * fy, 0, Tabla[1, np.argmin(np.abs(c - 0.6 * h))]],
            "Profundidad c (cm)": [h, 0,"N/A", 0.6 * d]
        }
    df_puntos = pd.DataFrame(puntos)
    st.subheader("📌 Puntos importantes")
    st.dataframe(df_puntos)






    # 📥 Descargar CSV del diagrama base
    df = pd.DataFrame({
        "Momento (kg-cm)": M,
        "Carga axial (kg)": P,
        "Eje neutro (cm)": c
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv, file_name="diagrama.csv", mime="text/csv")

else:
    st.warning("❌ No hay datos generados. Ve a una sección y presiona 'Generar Diagrama'")

