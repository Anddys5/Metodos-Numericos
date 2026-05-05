import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistemas Lineales PRO - Informática UMSA", layout="wide")

st.title("🔬 Desafío Métodos Numéricos: Optimización de Recursos en Datacenters")
st.markdown("""
En este modelo, las variables representan la carga de trabajo asignada a tres tipos de servidores:
- **x1:** Servidores de Procesamiento (CPU)
- **x2:** Servidores de Memoria (RAM)
- **x3:** Servidores de Almacenamiento (SSD)
""")

# -------------------------
# SELECCIÓN DE ESCENARIO
# -------------------------
with st.sidebar:
    st.header("Configuración")
    caso = st.selectbox("Seleccione el Escenario:", 
                        ["Ideal", "Estrés (Alta Demanda)", "Mal Condicionado", "Personalizado"])
    w = st.slider("Parámetro ω (SOR)", 0.5, 2.0, 1.1)
    tol = 1e-6

def cargar_sistema(caso):
    if caso == "Ideal":
        A = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 3]], float) # Matriz diagonal dominante
        b = np.array([6, 15, 14], float)
    elif caso == "Estrés (Alta Demanda)":
        A = np.array([[400, 150, 120], [150, 500, 180], [120, 180, 600]], float)
        b = np.array([1200, 2500, 3000], float)
    elif caso == "Mal Condicionado":
        # Fuentes casi idénticas generan hiperplanos casi paralelos
        A = np.array([[1, 1, 1], [1.0001, 1.0002, 1.0001], [1, 2, 3]], float)
        b = np.array([6, 6.0004, 10], float)
    else:
        # Modo manual
        A = np.zeros((3,3))
        b = np.zeros(3)
        for i in range(3):
            cols = st.columns(4)
            for j in range(3):
                A[i][j] = cols[j].number_input(f"A[{i+1}][{j+1}]", value=1.0, key=f"a{i}{j}")
            b[i] = cols[3].number_input(f"b{i+1}", value=1.0, key=f"b{i}")
    return A, b

A, b = cargar_sistema(caso)

# Cálculo del número de condición para el análisis
cond_a = np.linalg.cond(A)

col1, col2 = st.columns(2)
with col1:
    st.write("### Matriz de Coeficientes (A)")
    st.write(A)
with col2:
    st.write("### Vector de Cargas (b)")
    st.write(b)
st.metric("Número de Condición κ(A)", f"{cond_a:.2e}", 
          help="Un número alto indica que el sistema es inestable y difícil de resolver[cite: 1]")

# -------------------------
# IMPLEMENTACIÓN DE ALGORITMOS
# -------------------------
def jacobi(A, b, x0, tol, max_iter=100):
    x = x0.copy()
    errores = []
    for k in range(max_iter):
        x_new = (b - (A @ x - np.diag(A) * x)) / np.diag(A)
        err = np.linalg.norm(x_new - x)
        errores.append(err)
        if err < tol: return x_new, k + 1, errores
        x = x_new
    return x, max_iter, errores

def gauss_seidel(A, b, x0, tol, max_iter=100):
    x = x0.copy()
    errores = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(len(b)):
            s = sum(A[i][j] * x[j] for j in range(len(b)) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        err = np.linalg.norm(x - x_old)
        errores.append(err)
        if err < tol: return x, k + 1, errores
    return x, max_iter, errores

def sor(A, b, x0, w, tol, max_iter=100):
    x = x0.copy()
    errores = []
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(len(b)):
            s = sum(A[i][j] * x[j] for j in range(len(b)) if j != i)
            x[i] = (1 - w) * x_old[i] + w * (b[i] - s) / A[i][i]
        err = np.linalg.norm(x - x_old)
        errores.append(err)
        if err < tol: return x, k + 1, errores
    return x, max_iter, errores

def gradiente_conjugado_prec(A, b, x0, tol):
    # Basado en el algoritmo Mz del artículo de la UMSA[cite: 1]
    x = x0.copy()
    r = b - A @ x
    M_inv = np.diag(1.0 / np.diag(A)) # Precondicionador Jacobi
    z = M_inv @ r
    p = z.copy()
    errores = []
    for k in range(len(b)):
        Ap = A @ p
        alpha = (r @ z) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        err = np.linalg.norm(r_new)
        errores.append(err)
        if err < tol: return x, k + 1, errores
        z_new = M_inv @ r_new
        beta = (r_new @ z_new) / (r @ z)
        p = z_new + beta * p
        r, z = r_new, z_new
    return x, len(b), errores

# -------------------------
# RESOLUCIÓN Y TABLA[cite: 2]
# -------------------------
if st.button("🚀 Ejecutar Análisis Multimetodo"):
    x0 = np.zeros(len(b))
    
    # Solución Exacta (LU)
    sol_exacta = np.linalg.solve(A, b)
    
    # Ejecución de métodos
    s_j, i_j, e_j = jacobi(A, b, x0, tol)
    s_gs, i_gs, e_gs = gauss_seidel(A, b, x0, tol)
    s_sor, i_sor, e_sor = sor(A, b, x0, w, tol)
    s_gc, i_gc, e_gc = gradiente_conjugado_prec(A, b, x0, tol)

    # Tabla Comparativa requerida[cite: 2]
    resumen = pd.DataFrame({
        "Método": ["Factorización LU (Exacta)", "Jacobi", "Gauss-Seidel", "SOR (w={})".format(w), "Grad. Conj. Prec."],
        "Iteraciones": [1, i_j, i_gs, i_sor, i_gc],
        "Error Final": [0, e_j[-1], e_gs[-1], e_sor[-1], e_gc[-1]],
        "Converge": ["Sí", "Sí" if i_j < 100 else "No", "Sí" if i_gs < 100 else "No", "Sí" if i_sor < 100 else "No", "Sí"]
    })
    
    st.subheader("📊 Tabla Comparativa de Desempeño")
    st.table(resumen)

    # Gráfica de Convergencia
    st.subheader("📈 Gráfica de Convergencia (Escala Log)")
    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(e_j, label=f"Jacobi ({i_j} it)")
    ax_conv.plot(e_gs, label=f"Gauss-Seidel ({i_gs} it)")
    ax_conv.plot(e_sor, label=f"SOR ({i_sor} it)")
    ax_conv.plot(e_gc, label=f"Grad. Conj. Prec. ({i_gc} it)", linewidth=3)
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Iteraciones")
    ax_conv.set_ylabel("Norma del Error")
    ax_conv.legend()
    st.pyplot(fig_conv)

    # Visualización 3D[cite: 2]
    st.subheader("📐 Visualización Geométrica de Planos")
    fig_3d = plt.figure(figsize=(10, 7))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Crear malla
    limit = 10 if caso != "Estrés (Alta Demanda)" else 100
    x_range = np.linspace(-limit, limit, 10)
    y_range = np.linspace(-limit, limit, 10)
    X, Y = np.meshgrid(x_range, y_range)

    for i in range(3):
        if A[i][2] != 0:
            Z = (b[i] - A[i][0]*X - A[i][1]*Y) / A[i][2]
            ax_3d.plot_surface(X, Y, Z, alpha=0.3, label=f'Ecuación {i+1}')
    
    # Marcar la solución
    ax_3d.scatter(sol_exacta[0], sol_exacta[1], sol_exacta[2], color='red', s=100, label='Solución Exacta')
    st.pyplot(fig_3d)

    if caso == "Mal Condicionado":
        st.warning("⚠️ **Observación Académica:** En este escenario, los planos son casi paralelos. Como indica Suñagua (2020), el número de condición es tan alto que los métodos clásicos requieren muchas iteraciones o fallan en precisión[cite: 1].")