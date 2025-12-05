import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Medisonic Analytics", layout="wide")

# --- 1. CARGA DE DATOS DESDE EXCEL ---
st.sidebar.header("📂 Fuente de Datos")

# Función para cargar datos sin errores
@st.cache_data
def cargar_datos():
    try:
        # Busca el archivo en la misma carpeta
        df = pd.read_excel("data_medisonic.xlsx")
        return df
    except FileNotFoundError:
        return None

df_raw = cargar_datos()

if df_raw is None:
    st.error("⚠️ NO SE ENCUENTRA EL ARCHIVO 'data_medisonic.xlsx'")
    st.warning("Por favor crea un Excel con columnas: AÑO, TRIMESTRE, CLIENTE, CATEGORIA, MONTO y guárdalo en la misma carpeta.")
    st.stop() # Detiene la ejecución si no hay datos

# --- 2. BARRA LATERAL (FILTROS AVANZADOS) ---
st.sidebar.subheader("⚙️ Filtros del Modelo")

# Filtro de Años
anos_disponibles = sorted(df_raw['AÑO'].unique())
anos_selec = st.sidebar.multiselect("Seleccionar Años para Análisis", anos_disponibles, default=anos_disponibles)

# Filtro de Categoría (Para quitar Megaproyectos)
cats_disponibles = df_raw['CATEGORIA'].unique()
cats_selec = st.sidebar.multiselect("Categorías a incluir", cats_disponibles, default=cats_disponibles)

# Aplicar filtros
df = df_raw[(df_raw['AÑO'].isin(anos_selec)) & (df_raw['CATEGORIA'].isin(cats_selec))]

# --- 3. TÍTULO Y PESTAÑAS ---
st.title("📊 Dashboard Gerencial - Medisonic 2026")
tab1, tab2, tab3 = st.tabs(["📈 Proyección & Estrategia", "🔍 Auditoría de Datos", "📋 Tabla Detallada"])

# --- PESTAÑA 1: PROYECCIÓN ---
with tab1:
    # KPIs
    ventas_por_anio = df.groupby('AÑO')['MONTO'].sum().reset_index()
    if not ventas_por_anio.empty:
        venta_actual = ventas_por_anio.iloc[-1]['MONTO']
        anio_actual = ventas_por_anio.iloc[-1]['AÑO']
        promedio_venta = df['MONTO'].mean()
    else:
        venta_actual = 0
        anio_actual = 0
        promedio_venta = 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Ventas {anio_actual}", f"S/ {venta_actual:,.0f}")
    c2.metric("Total Registros Analizados", f"{len(df)}")
    c3.metric("Ticket Promedio", f"S/ {promedio_venta:,.0f}")

    st.markdown("---")

    # MODELO PREDICTIVO
    if len(ventas_por_anio) > 1:
        X = ventas_por_anio[['AÑO']]
        y = ventas_por_anio['MONTO']
        model = LinearRegression()
        model.fit(X, y)
        pred_2026 = model.predict([[2026]])[0]
        r2 = model.score(X, y)

        col_izq, col_der = st.columns([2, 1])
        
        with col_izq:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ventas_por_anio['AÑO'], y=ventas_por_anio['MONTO'],
                                     mode='lines+markers+text', name='Real',
                                     text=[f"{v/1e6:.1f}M" for v in ventas_por_anio['MONTO']],
                                     textposition="top center", line=dict(color='#0052cc', width=3)))
            fig.add_trace(go.Scatter(x=[2025, 2026], y=[venta_actual, pred_2026],
                                     mode='lines+markers+text', name='Proyección IA',
                                     text=["", f"{pred_2026/1e6:.2f}M"],
                                     textposition="top center",
                                     line=dict(color='#ff8800', width=3, dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_der:
            st.success(f"### Proyección 2026: \n # S/ {pred_2026:,.0f}")
            st.info(f"""
            **Calidad del Ajuste (R²): {r2:.2f}**
            * Si R² es bajo (<0.5), significa que las ventas son muy inestables y la proyección es referencial.
            * Si R² es alto (>0.8), la tendencia es sólida.
            """)

# --- PESTAÑA 2: AUDITORÍA DE DATOS (NUEVO) ---
with tab2:
    st.header("Análisis de la Situación de Datos")
    st.markdown("Utiliza esta sección para entender la dispersión y detectar errores en la carga de información.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Distribución de Ventas (Boxplot)")
        st.markdown("*Ayuda a ver los rangos de precios y detectar outliers (puntos alejados).*")
        fig_box = px.box(df, x="CATEGORIA", y="MONTO", points="all", color="CATEGORIA",
                         title="Dispersión de Precios por Categoría")
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        st.subheader("Peso de Clientes")
        ventas_cliente = df.groupby('CLIENTE')['MONTO'].sum().reset_index().sort_values('MONTO', ascending=False).head(10)
        fig_barh = px.bar(ventas_cliente, x='MONTO', y='CLIENTE', orientation='h',
                          title="Top 10 Clientes (Pareto)", text_auto='.2s')
        st.plotly_chart(fig_barh, use_container_width=True)

# --- PESTAÑA 3: TABLA ---
with tab3:
    st.subheader("Base de Datos Procesada")
    st.dataframe(df, use_container_width=True)