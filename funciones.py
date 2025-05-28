import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import math
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import sympy as sp
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
# --- FUNCIÓN PRINCIPAL PARA RECTANGULAR ---
def diagrama_interaccion(As, b, h, r, fy, fpc):
    As=np.array(As)
    d=h-r
    E_acero = 2100000
    epd = 0.85 * fpc
    dist = (h - 2 * r) / (len(As) - 1)
    distancias = np.array([i * dist + r for i in range(len(As))])
    
    # Advertencia si hay lechos separados más de 15 cm
    lechos_excedidos = []
    for j in range(len(distancias) - 1):
        diferencia = distancias[j + 1] - distancias[j]
        if diferencia > 15:
            lechos_excedidos.append((j + 1, diferencia))
    if lechos_excedidos:
        mensaje = "⚠️ Ojo, hay lechos con más de 15 cm de separación:\n"
        for indice, dif in lechos_excedidos:
            mensaje += f"- Entre lecho {indice} y lecho {indice + 1}: {dif:.2f} cm\n"
        st.warning(mensaje)

    aconc = b * h - np.sum(As)
    F_conc = epd * aconc
    F_As = np.array(As) * fy
    F_total = np.sum(F_As) + F_conc
    Mo = np.sum(F_As * (h - distancias)) + F_conc * (h / 2)
    cp = h - Mo / F_total

    c = np.arange(h - 1, 0, -0.5)
    #Calculo de beta1 para el bloque a compresion
    if fpc>=170 and fpc<=280:
        beta1=0.85
    elif fpc>280 and fpc<550:
        beta1=0.85-(0.05*(fpc-28)/7)
    else:
        beta1=0.65
    a = beta1 * c

    A = []
    for j in c:
        suma_areas = np.sum(As[distancias <= j])
        A.append((j, suma_areas))
    A = np.array(A)

    A_concreto_efectiva = a * b - A[:, 1]
    P_concreto = A_concreto_efectiva * epd

    es = np.array([[0.003 * ((c_i - d_j) / c_i) for d_j in distancias] for c_i in c])
    esf_acero = np.clip(es * E_acero, -fy, fy)
    P_acero = esf_acero * As

    P = P_acero.sum(axis=1) + P_concreto
    M = np.sum(P_acero * (cp - distancias), axis=1) + np.sum(P_concreto * (cp - a / 2), axis=0)

    return M, P, c, F_total,d,h
# --- AJUSTE POLINOMIAL ---
def ajustar_modeloM(P, M):
    mejor_r2 = -1
    mejor_modelo = None
    mejor_coef = None
    for i in range(1, 10):
        coef = np.polyfit(P, M, i)
        p = np.poly1d(coef)
        M_predicho = p(P)
        r2 = r2_score(M, M_predicho)
        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_modelo = p
            mejor_coef = coef
    return mejor_modelo, mejor_r2, mejor_coef
##  Diagrama Circular
def diagrama_interaccion_circular(As, D, r, fy, fpc):
    #Librerias necesarias
    As=np.array(As)
    h=D
    d = D - r
    # Calculo de las distancias de los lechos de acero
    E_acero = 2100000
    epd = 0.85 * fpc
    dist = (h - 2 * r) / (len(As) - 1)
    d=h-r
    
    distancias = []  # Lista vacía para acumular las distancias
    for i in range(len(As)):
        valor = i * dist + r
        distancias.append(valor)
    distancias = np.array(distancias)
    
    # Verificación de la distancia mínima entre lechos
    for j in range(len(distancias) - 1):
        diferencia = distancias[j + 1] - distancias[j]
        if diferencia > 15:
            print("Ojo, la distancia máxima entre lechos es de 15 cm")

    #Calculo del centroide platico
    aconc=((math.pi*h**2)/4)-np.sum(As)#Area efectiva de concreto
    F_conc=epd*aconc#Fuerza del concreto
    F_As=As*fy#Fuerza del acero
    F_total=np.sum(F_As)+F_conc
    Mo=np.sum(F_As*(d-distancias))+F_conc*(d-h/2)
    cp=d-Mo/F_total
        #Calculo de beta1 para el bloque a compresion
    if fpc>=170 and fpc<=280:
        beta1=0.85
    elif fpc>280 and fpc<550:
        beta1=0.85-(0.05*(fpc-28)/7)
    else:
        beta1=0.65
    # Calculo de un vector que da el área de acero acumulada para el cálculo del área de concreto efectiva
    c = np.arange(h - 1, 0, -0.5)  # Profundidad del eje neutro
    a = beta1 * c  # Bloque a compresión de concreto
    A = []
    for j in c:
        suma_areas = 0
        indice_coincidencias = np.where(np.abs(distancias <= j))[0]
        if indice_coincidencias.size > 0:
            for i in indice_coincidencias:
                suma_areas += As[i]
        A.append((j, suma_areas))
    A = np.array([A]).squeeze()
    
    #Calculamos el area de concreto efectiva y la fuerza
    A_concreto_bruta=[]
    y_test_concreto=[]
    for i in range(len(a)):
        if a[i]<=h/2:
            A_concreto=((h/2)**2 * np.arccos(((h/2) - a[i]) / (h/2)) - ((h/2) - a[i]) * np.sqrt(2 * (h/2) * a[i] - a[i]**2))
        elif a[i]>h/2:
            A_concreto= np.pi*(h/2)**2-((h/2)**2 * np.arccos((a[i] - (h/2)) / (h/2)) - (a[i] - (h/2)) * np.sqrt(2 * (h/2) * a[i] - a[i]**2))
        else:
            A_concreto=np.pi*(h/2)**2
        A_concreto_bruta.append(A_concreto)

    for j in range(len(c)):
            # Para calcular el centroide, evitamos la división por cero
            if c[j] == 0:
                y_inf = 0
            else:
                theta = 2 * np.arccos(((h / 2) - c[j]) / (h / 2))
                seno = np.sin(theta / 2)
                y_inf = (4 * (h / 2) * seno ** 3) / (3 * (theta - np.sin(theta)))  # Centroide del área a compresión desde abajo
            # Evitamos problemas cuando c[j] es igual a h (superior del círculo)
            if c[j] == h:
                y_sup = 0  # El centroide está en la base en este caso
            else:
                y_sup = (h / 2) - y_inf
            y_test_concreto.append(y_sup)
    A_concreto_efectiva=np.array([A_concreto_bruta])-A[:,1]
    P_concreto=A_concreto_efectiva*epd

    
    # Cálculo de las deformaciones en el acero
    deformaciones = []  # Lista que almacenará las deformaciones para cada c
    for i in range(len(A)):
        for j in range(len(distancias)):
            es = 0.003 * ((c[i] - distancias[j]) / c[i])
            deformaciones.append(es)
    es = np.array([deformaciones]).squeeze()
    es = es.reshape(len(c), len(As))
    
    # Cálculo de las fuerzas en el acero
    esf_acero = es * E_acero  # Calcula el esfuerzo del acero
    esf_acero = np.clip(esf_acero, -fy, fy)  # Limita el esfuerzo a fy
    P_acero = esf_acero * As
    P_acero = np.array([P_acero]).squeeze()
    print(cp-distancias)
    # Cálculo de P para cada punto
    P = P_acero.sum(axis=1) + P_concreto
    P=np.array(P).flatten()
    
    # Ahora que tenemos las distancias y las fuerzas, podemos calcular Mo con las dimensiones correctas
    
    M=np.sum(P_acero*(cp-distancias),axis=1)+np.sum(P_concreto*(cp-a/2))
    M=np.array(M).flatten()
    
    return M,P,c,F_total,d,h
def diagrama_interaccion_cualquiera(As, verts, h, r, fy, fpc):
        # Calculo de las distancias de los lechos de acero
        As=np.array(As)
        E_acero = 2100000
        epd = 0.85 * fpc
        dist = (h - 2 * r) / (len(As) - 1)
        d=h-r
        
        distancias = []  # Lista vacía para acumular las distancias
        for i in range(len(As)):
            valor = i * dist + r
            distancias.append(valor)
        distancias = np.array(distancias)
        print(distancias)
        
        # Verificación de la distancia mínima entre lechos
        for j in range(len(distancias) - 1):
            diferencia = distancias[j + 1] - distancias[j]
            if diferencia > 15:
                print("Ojo, la distancia máxima entre lechos es de 15 cm")
                    #Calculo de beta1 para el bloque a compresion
        if fpc>=170 and fpc<=280:
            beta1=0.85
        elif fpc>280 and fpc<550:
            beta1=0.85-(0.05*(fpc-28)/7)
        elif fpc<170:
            print("No debe usarse ese tipo de concreto,al menos 250kg/cm2")
        else:
            beta1=0.65
            #Fucniones para calcular el area y centroide de la seccion
        def Area_concreto(verts):
                n = len(verts)
                area = 0
                for i in range(n):
                    j = (i + 1) % n
                    x1, y1 = verts[i]
                    x2, y2 = verts[j]
                    area += (x1 * y2 - x2 * y1)
                return abs(area) / 2
        def y_conc(verts):
                n = len(verts)
                A = Area_concreto(verts)
                Cy = 0
                for i in range(n):
                    j = (i + 1) % n
                    x1, y1 = verts[i]
                    x2, y2 = verts[j]
                    common = (x1 * y2 - x2 * y1)
                    Cy += (y1 + y2) * common
                return Cy / (6 * A)

                #Calculo del centroide platico
        #Calculo del centroide platico
        Area_efectiva=Area_concreto(verts)-np.sum(As)#Area efectiva de concreto
        F_conc=epd*Area_efectiva#Fuerza del concreto
        F_As=As*fy#Fuerza del acero
        F_total=np.sum(F_As)+F_conc
        Mo=np.sum(F_As*(d-distancias))+F_conc*(d-y_conc(verts))
        cp=d-Mo/F_total

        # Calculo de un vector que da el área de acero acumulada para el cálculo del área de concreto efectiva
        c = np.arange(h - 1, 0, -0.5)  # Profundidad del eje neutro
        a = beta1 * c  # Bloque a compresión de concreto
        A = []
        for j in c:
            suma_areas = 0
            indice_coincidencias = np.where(np.abs(distancias <= j))[0]
            if indice_coincidencias.size > 0:
                for i in indice_coincidencias:
                    suma_areas += As[i]
            A.append((j, suma_areas))
        A = np.array([A]).squeeze()
            #Funcion del Calculo del area de concreto a compresion y su centroide
        def A_concreto_compresion(polygon_coords, h, altura_total=60.0):

            poly = Polygon(polygon_coords)
            minx, miny, maxx, maxy = poly.bounds

            if h >= maxy:
                return 0.0, None  # No hay compresión
            elif h <= miny:
                area = poly.area
                yc = poly.centroid.y
                return area, altura_total - yc  # Centroide desde la base
            else:
                rect = box(minx - 1, h, maxx + 1, maxy + 1)
                interseccion = poly.intersection(rect)

                if interseccion.is_empty:
                    return 0.0, None
                elif interseccion.geom_type == 'Polygon':
                    area = interseccion.area
                    yc = interseccion.centroid.y
                    return area, altura_total - yc
                elif interseccion.geom_type == 'MultiPolygon':
                    area = sum(p.area for p in interseccion.geoms)
                    yc_total = sum(p.area * p.centroid.y for p in interseccion.geoms) / area
                    return area, altura_total - yc_total
                else:
                    return 0.0, None
        # Calculamos el área comprimida bruta y su centroide para todas las areas
        resultados = []

        for i in a:
                area, y_c= A_concreto_compresion(verts, h - i,h)
                diferencia = h - y_c  
                resultados.append([area, diferencia])
        resultados = np.array(resultados)
        y_test_concreto=resultados[:, 1]#Centroide
        # Ahora podemos restar las columnas
        A_concreto_efectiva = resultados[:, 0] - A[:, 1]

        #Fuerza del concreto
        P_concreto=epd*A_concreto_efectiva
        
        # Cálculo de las deformaciones en el acero
        deformaciones = []  # Lista que almacenará las deformaciones para cada c
        for i in range(len(A)):
            for j in range(len(distancias)):
                es = 0.003 * ((c[i] - distancias[j]) / c[i])
                deformaciones.append(es)
        es = np.array([deformaciones]).squeeze()
        es = es.reshape(len(c), len(As))
        
        # Cálculo de las fuerzas en el acero
        esf_acero = es * E_acero  # Calcula el esfuerzo del acero
        esf_acero = np.clip(esf_acero, -fy, fy)  # Limita el esfuerzo a fy
        P_acero = esf_acero * As
        P_acero = np.array([P_acero]).squeeze()
        print(cp-distancias)
        # Cálculo de P para cada punto
        P = P_acero.sum(axis=1) + P_concreto
        
        # Ahora que tenemos las distancias y las fuerzas, podemos calcular Mo con las dimensiones correctas
        
        M=np.sum(P_acero*(cp-distancias),axis=1)+np.sum(P_concreto*(cp-y_test_concreto))
        
        return M,P,c,F_total,d,h
def predecir_c(M,P,c):
    
    X = np.column_stack((P, M))
    mejor_r2 = -1
    mejor_grado = None
    mejor_modelo = None
    mejor_transformador = None

    for grado in range(1, 10):
        poly = PolynomialFeatures(degree=grado, include_bias=False)
        X_poly = poly.fit_transform(X)

        modelo = LinearRegression()
        modelo.fit(X_poly, c)
        c_predicho = modelo.predict(X_poly)
        r2 = r2_score(c, c_predicho)

        if r2 > mejor_r2:
            mejor_r2 = r2
            mejor_grado = grado
            mejor_modelo = modelo
            mejor_transformador = poly

    return mejor_modelo, mejor_transformador, mejor_r2, mejor_grado



def ajustar_modeloP(M, P):
    if len(M) < 3:
        return None, 0, None

    mejor_r2 = -1
    mejor_modelo = None
    mejor_coef = None
    max_grado = min(5, len(M) - 1)  # evita overfitting

    for i in range(1, max_grado + 1):
        try:
            coef = np.polyfit(M, P, i)
            p = np.poly1d(coef)
            P_predicho = p(M)
            r2 = r2_score(P, P_predicho)
            if r2 > mejor_r2:
                mejor_r2 = r2
                mejor_modelo = p
                mejor_coef = coef
        except:
            continue
    return mejor_modelo, mejor_r2, mejor_coef

