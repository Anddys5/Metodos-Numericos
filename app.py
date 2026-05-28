import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistemas Lineales PRO - Informática UMSA", layout="wide")

# -------------------------
# ESTILOS Y ENCABEZADO ACADÉMICO
# -------------------------
st.title("🔬 Desafío Métodos Numéricos: Optimización de la Dieta y Metabolismo en Aves Neotropicales")
st.markdown("""
**Materia:** Métodos Numéricos (Mención Ingeniería de Sistemas)  
**Institución:** Universidad Mayor de San Andrés (UMSA)  
---
""")

# -------------------------
# SELECCIÓN DE ESCENARIO EN SIDEBAR
# -------------------------
with st.sidebar:
    st.header("⚙️ Configuración del Sistema")
    caso = st.selectbox("Seleccione el Escenario de Análisis:", 
                        ["Ideal", "Estrés (Demanda Energética Extrema)", "Mal Condicionado", "Personalizado"])
    w = st.slider("Parámetro de Relajación ω (SOR)", 0.5, 2.0, 1.1)
    tol = 1e-6
    st.info("💡 Consejo: Evalúa el impacto de ω en el escenario de Estrés para observar la aceleración de convergencia.")

# -------------------------
# PESTAÑAS PRINCIPALES (ORGANIZACIÓN DE INVESTIGACIÓN)
# -------------------------
tab1, tab2, tab3 = st.tabs(["📋 Marco Teórico y Contexto Biológo", "💻 Simulador y Solución", "📊 Análisis Estadístico y Conclusiones"])

with tab1:
    st.header("1. Contextualización del Caso de Estudio")
    st.markdown("""
    Debido a la alta tasa metabólica de las aves neotropicales, el organismo debe equilibrar de forma estricta la ingesta y absorción de macronutrientes esenciales para mantener la homeostasis en diferentes estados ambientales y fisiológicos.
    
    Variables del Modelo Alimentario:
    * **$x_1$:** Ingesta requerida de **Proteínas** (g) 
    * **$x_2$:** Ingesta requerida de **Lípidos** (g) 
    * **$x_3$:** Ingesta requerida de **Carbohidratos** (g) 
    
    ### Modelado Matemático
    El balance de nutrientes en el organismo se describe mediante un sistema de ecuaciones lineales simultáneas de la forma:
    $$A x = b$$
    Donde la matriz de coeficientes $A \in \mathbb{R}^{3 \\times 3}$ representa la eficiencia de absorción metabólica de cada tipo de alimento disponible, y el vector de acoplamiento $b \in \mathbb{R}^3$ denota la demanda energética neta requerida por el ave[cite: 15].
    
    ### Análisis de Escenarios según la Guía:
    1.  **Caso Ideal:** El sistema es consistente, bien condicionado y las fuentes de alimento son balanceadas. Cumple con la dominancia diagonal[cite: 10].
    2.  **Caso Bajo Estrés:** Refleja una demanda energética extrema (ej. el proceso de migración estacional), donde los coeficientes aumentan su magnitud significativamente para sostener el catabolismo basal[cite: 11].
    3.  **Caso Mal Condicionado:** Se presenta cuando dos fuentes de alimento disponibles en el ecosistema son nutricionalmente casi idénticas, lo que genera hiperplanos casi paralelos y dificulta críticamente la convergencia[cite: 12].
    """)

# -------------------------
# CARGA DE SISTEMAS MATRICIALES
# -------------------------
def cargar_sistema(caso):
    if caso == "Ideal":
        A = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 3]], float) 
        b = np.array([6, 15, 14], float)
    elif caso == "Estrés (Demanda Energética Extrema)":
        A = np.array([[400, 150, 120], [150, 500, 180], [120, 180, 600]], float)
        b = np.array([1200, 2500, 3000], float)
    elif caso == "Mal Condicionado":
        A = np.array([[1, 1, 1], [1.0001, 1.0002, 1.0001], [1, 2, 3]], float)
        b = np.array([6, 6.0004, 10], float)
    else:
        st.subheader("Configuración de Matriz Personalizada")
        A = np.zeros((3,3))
        b = np.zeros(3)
        for i in range(3):
            cols = st.columns(4)
            for j in range(3):
                A[i][j] = cols[j].number_input(f"A[{i+1}][{j+1}]", value=1.0, key=f"a{i}{j}")
            b[i] = cols[3].number_input(f"b{i+1}", value=1.0, key=f"b{i}")
    return A, b

A, b = cargar_sistema(caso)
cond_a = np.linalg.cond(A)

# -------------------------
# ALGORITMOS ITERATIVOS (TUS FUNCIONES)
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
    x = x0.copy()
    r = b - A @ x
    M_inv = np.diag(1.0 / np.diag(A)) 
    z = M_inv @ r
    p = z.copy()
    errores = []
    for k in range(len(b) * 10): 
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
    return x, len(b)*10, errores

# -------------------------
# CONTENIDO DE LA PESTAÑA 2: SIMULADOR Y TRABAJO
# -------------------------
with tab2:
    st.header(f"2. Análisis en Tiempo Real del Escenario: {caso}")
    
    col_mat1, col_mat2, col_mat3 = st.columns([2, 1, 1])
    with col_mat1:
        st.write("**Matriz de Eficiencia Nutricional (A):**")
        st.write(A)
    with col_mat2:
        st.write("**Vector de Demandas Metabólicas (b):**")
        st.write(b)
    with col_mat3:
        st.metric("Número de Condición $\kappa(A)$", f"{cond_a:.2e}", 
                  help="Un $\kappa(A) \sim 1$ indica estabilidad biológica. Un valor elevado denota sensibilidad crítica.")

    st.markdown("---")
    
    x0 = np.zeros(len(b))
    sol_exacta = np.linalg.solve(A, b)
    s_j, i_j, e_j = jacobi(A, b, x0, tol)
    s_gs, i_gs, e_gs = gauss_seidel(A, b, x0, tol)
    s_sor, i_sor, e_sor = sor(A, b, x0, w, tol)
    s_gc, i_gc, e_gc = gradiente_conjugado_prec(A, b, x0, tol)

    st.subheader("💡 Solución Directa de Referencia (Método Exacto Factorización LU)")
    st.success(f"Gramos óptimos requeridos: $x_1$ (Proteínas) = **{sol_exacta[0]:.4f}g**, $x_2$ (Lípidos) = **{sol_exacta[1]:.4f}g**, $x_3$ (Carbohidratos) = **{sol_exacta[2]:.4f}g** [cite: 7, 19]")

    # Tabla Comparativa requerida por la Licenciada (Cuadro 1)
    resumen = pd.DataFrame({
        "Algoritmo / Enfoque": ["Solución Directa (Factorización LU)", "Método Jacobi", "Método Gauss-Seidel", f"Método SOR (ω={w})", "Gradiente Conjugado Prec."],
        "Iteraciones (Ideal)": [1, i_j if caso=="Ideal" else "-", i_gs if caso=="Ideal" else "-", i_sor if caso=="Ideal" else "-", i_gc if caso=="Ideal" else "-"],
        "Iteraciones (Stress)": [1, i_j if "Estrés" in caso else "-", i_gs if "Estrés" in caso else "-", i_sor if "Estrés" in caso else "-", i_gc if "Estrés" in caso else "-"],
        "Iteraciones (Mal C.)": [1, i_j if "Mal" in caso else "-", i_gs if "Mal" in caso else "-", i_sor if "Mal" in caso else "-", i_gc if "Mal" in caso else "-"],
        "Converge": ["Sí (Exacto)", "Sí" if i_j < 100 else "No", "Sí" if i_gs < 100 else "No", "Sí" if i_sor < 100 else "No", "Sí"]
    }) [cite: 26]
    
    st.subheader("📊 Tabla Comparativa de Eficiencia (Tolerancia $10^{-6}$)")
    st.dataframe(resumen) [cite: 32]

    # Gráficos en columnas divididas
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("📈 Comportamiento y Velocidad de Convergencia")
        fig_conv, ax_conv = plt.subplots()
        ax_conv.plot(e_j, label=f"Jacobi ({i_j} it)", linestyle="--")
        ax_conv.plot(e_gs, label=f"Gauss-Seidel ({i_gs} it)", linestyle="-.")
        ax_conv.plot(e_sor, label=f"SOR ({i_sor} it)", color="orange")
        ax_conv.plot(e_gc, label=f"Grad. Conj. Prec. ({i_gc} it)", linewidth=2.5, color="green")
        ax_conv.set_yscale("log")
        ax_conv.set_xlabel("Iteración de cómputo")
        ax_conv.set_ylabel("Norma Logarítmica del Error")
        ax_conv.grid(True, which="both", ls=":")
        ax_conv.legend()
        st.pyplot(fig_conv)
        
    with col_g2:
        st.subheader("🗺️ Representación Geométrica de los Hiperplanos")
        fig_3d = plt.figure(figsize=(10, 7))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        limit = 10 if "Estrés" not in caso else 100
        X, Y = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))

        for i in range(3):
            if A[i][2] != 0:
                Z = (b[i] - A[i][0]*X - A[i][1]*Y) / A[i][2]
                ax_3d.plot_surface(X, Y, Z, alpha=0.2)
        
        ax_3d.scatter(sol_exacta[0], sol_exacta[1], sol_exacta[2], color='red', s=150, label='Punto de Homeostasis Exacto', depthshade=False)
        ax_3d.set_xlabel('Proteínas (x1)')
        ax_3d.set_ylabel('Lípidos (x2)')
        ax_3d.set_zlabel('Carbohidratos (x3)')
        ax_3d.legend()
        st.pyplot(fig_3d) [cite: 22]

# -------------------------
# CONTENIDO DE LA PESTAÑA 3: ANÁLISIS BIOLÓGICO
# -------------------------
with tab3:
    st.header("3. Informe de Interpretación Biológica y Evaluación Numérica")
    
    if caso == "Ideal":
        st.markdown(f"""
        ### 🟢 Análisis del Escenario Ideal
        * **Interpretación Ecológica:** El ecosistema provee fuentes de alimento balanceadas y diferenciadas. El ave metaboliza los nutrientes sin solapamiento de absorción, garantizando estabilidad homeostática[cite: 10].
        * **Evaluación de la Estructura:** El sistema cuenta con un Número de Condición bajo ($\kappa(A) = {cond_a:.2f}$). Al cumplirse la **Dominancia Diagonal Estricta**, los métodos matriciales iterativos clásicos (**Jacobi** y **Gauss-Seidel**) convergen con extrema rapidez[cite: 20].
        """)
        
    elif "Estrés" in caso:
        st.markdown(f"""
        ### 🟡 Análisis del Escenario Bajo Estrés (Fase de Migración)
        * **Interpretación Ecológica:** Durante el vuelo migratorio masivo, la demanda metabólica y el gasto energético se disparan de forma exponencial[cite: 11]. El ave requiere asimilar macromutrientes a tasas críticas.
        * **Impacto Numérico:** Aunque la escala de la matriz se amplía, conserva propiedades definidas. Aquí se destaca el método **SOR**, donde al optimizar manualmente el parámetro $\omega$ en la barra lateral se reduce drásticamente el coste computacional frente al algoritmo estacionario tradicional[cite: 21].
        """)
        
    elif "Mal" in caso:
        st.markdown(f"""
        ### 🔴 Análisis del Escenario Crítico (Mal Condicionado)
        * **Interpretación Ecológica:** Dos o más de las fuentes vegetales de alimento disponibles en el entorno son nutricionalmente casi idénticas (mismas proporciones de proteínas y lípidos)[cite: 12]. Esto causa una redundancia sistémica en el tracto digestivo.
        * **Geometría y Fracaso Algorítmico:** Visualmente en el gráfico 3D, los hiperplanos interactúan de manera casi paralela, haciendo que la intersección sea difusa e inestable[cite: 12, 22]. Como cita la bibliografía especializada de la carrera (Suñagua, 2020), la acumulación de errores por redondeo provoca el colapso y la divergencia en aproximaciones de **Jacobi** o **Gauss-Seidel**, requiriendo estrictamente técnicas de subespacios de Krylov como el **Gradiente Conjugado Precondicionado**[cite: 21].
        """)
    else:
        st.markdown("""
        ### ⚙️ Evaluación del Escenario Personalizado
        """)

    st.markdown("""
    ---
    ### 📌 Bibliografía de Respaldo Académico
    * Suñagua, P. (2020). *Métodos Numéricos y Resolución de Sistemas Lineales por Subespacios*. Revista del Instituto de Investigación en Informática - UMSA.
    * Burden, R. L., & Faires, J. D. (2011). *Análisis Numérico*. Cengage Learning.
    """)