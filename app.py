import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
import torch
import torch.nn as nn
from PIL import Image
import json
import io

# Importar módulos personalizados
from utils.palm_detector import PalmDetector, segment_individual_palms
from utils.time_series_model import PalmLSTM, predict_stress
from utils.visualization import (
    create_ndvi_timeseries_plot,
    create_palm_distribution_map,
    create_stress_heatmap
)

# Configuración de la página
st.set_page_config(
    page_title="Monitoreo Inteligente - Palma Aceitera",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar estilos CSS
def load_css():
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Título principal
st.title("🌴 Sistema de Monitoreo Inteligente - Palma Aceitera")
st.markdown("---")

# Sidebar para navegación
st.sidebar.image("assets/logo.png", width=150)
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox(
    "Selecciona el módulo",
    ["📊 Dashboard", "🛰️ Cargar Datos", "🔍 Detección de Palmeras", 
     "📈 Análisis Temporal", "⚠️ Alertas Tempranas", "🗺️ Mapa de Distribución"]
)

# Clase para manejar el estado de la sesión
class SessionState:
    def __init__(self):
        self.uploaded_images = []
        self.palm_data = []
        self.time_series_data = None
        self.distribution_analysis = None
        self.model = None

# Inicializar estado de sesión
if 'state' not in st.session_state:
    st.session_state.state = SessionState()

# Módulo 1: Dashboard
if app_mode == "📊 Dashboard":
    st.header("Dashboard de Monitoreo")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Palmas Monitoreadas",
            value="1,247",
            delta="+12 esta semana"
        )
    
    with col2:
        st.metric(
            label="Tasa de Estrés",
            value="8.2%",
            delta="-2.1%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Distancia Promedio",
            value="7.8m",
            delta="Óptima"
        )
    
    with col4:
        st.metric(
            label="NDVI Promedio",
            value="0.74",
            delta="+0.03"
        )
    
    # Gráficos del dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Salud por Lote")
        health_data = pd.DataFrame({
            'Estado': ['Óptimo', 'Saludable', 'Estrés Leve', 'Estrés Severo'],
            'Cantidad': [650, 450, 120, 27]
        })
        fig = px.pie(health_data, values='Cantidad', names='Estado')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tendencia NDVI (Últimas 12 semanas)")
        dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
        ndvi_trend = [0.72, 0.73, 0.74, 0.72, 0.71, 0.70, 0.69, 0.68, 0.70, 0.72, 0.73, 0.74]
        fig = px.line(x=dates, y=ndvi_trend, title="NDVI Semanal")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="NDVI")
        st.plotly_chart(fig, use_container_width=True)

# Módulo 2: Cargar Datos
elif app_mode == "🛰️ Cargar Datos":
    st.header("Carga de Datos Multiespectrales")
    
    uploaded_files = st.file_uploader(
        "Carga imágenes multiespectrales",
        type=['tif', 'tiff', 'png', 'jpg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.state.uploaded_images = uploaded_files
        st.success(f"{len(uploaded_files)} imágenes cargadas exitosamente")
        
        # Mostrar preview de imágenes
        st.subheader("Vista Previa de Imágenes")
        cols = st.columns(3)
        for idx, uploaded_file in enumerate(uploaded_files[:3]):
            with cols[idx]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Imagen {idx+1}", use_column_width=True)
    
    # Cargar datos temporales
    st.subheader("Datos de Series Temporales")
    ts_file = st.file_uploader("Cargar CSV de series temporales", type=['csv'])
    
    if ts_file:
        df = pd.read_csv(ts_file)
        st.session_state.state.time_series_data = df
        st.dataframe(df.head())
        
        st.subheader("Resumen Estadístico")
        st.write(df.describe())

# Módulo 3: Detección de Palmeras
elif app_mode == "🔍 Detección de Palmeras":
    st.header("Detección Individual de Palmeras")
    
    if not st.session_state.state.uploaded_images:
        st.warning("Por favor carga imágenes en el módulo 'Cargar Datos' primero.")
    else:
        # Parámetros de detección
        col1, col2 = st.columns(2)
        
        with col1:
            min_area = st.slider("Área mínima de detección (píxeles)", 10, 100, 50)
            sensitivity = st.slider("Sensibilidad de detección", 0.1, 1.0, 0.5)
        
        with col2:
            st.subheader("Procesar Imágenes")
            if st.button("Iniciar Detección de Palmeras"):
                with st.spinner("Procesando imágenes..."):
                    # Aquí iría la lógica de detección
                    # Simulación por ahora
                    simulated_palms = [
                        {'id': f'P{i:03d}', 'x': np.random.randint(100, 900), 
                         'y': np.random.randint(100, 900), 'area': np.random.randint(50, 200)}
                        for i in range(1, 46)
                    ]
                    st.session_state.state.palm_data = simulated_palms
                    
                    st.success(f"✅ {len(simulated_palms)} palmas detectadas")
        
        # Mostrar resultados
        if st.session_state.state.palm_data:
            st.subheader("Resultados de Detección")
            
            # Mapa de distribución
            df_palms = pd.DataFrame(st.session_state.state.palm_data)
            fig = px.scatter(df_palms, x='x', y='y', size='area', 
                           title="Distribución de Palmeras Detectadas",
                           hover_data=['id'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Palmeras", len(df_palms))
            with col2:
                st.metric("Área Promedio", f"{df_palms['area'].mean():.1f} px")
            with col3:
                st.metric("Densidad", f"{len(df_palms)/10000:.1f} palmas/ha")

# Módulo 4: Análisis Temporal
elif app_mode == "📈 Análisis Temporal":
    st.header("Análisis de Series Temporales")
    
    # Simular datos temporales si no hay datos cargados
    if st.session_state.state.time_series_data is None:
        st.info("Generando datos de ejemplo...")
        
        # Crear datos de ejemplo
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='W')
        palm_ids = [f'P{i:03d}' for i in range(1, 11)]
        
        data = []
        for palm_id in palm_ids:
            base_ndvi = np.random.normal(0.75, 0.05)
            for date in dates:
                # Simular variación temporal
                trend = 0.01 * (date - dates[0]).days / 7
                noise = np.random.normal(0, 0.02)
                seasonal = 0.03 * np.sin(2 * np.pi * (date - dates[0]).days / 30)
                
                ndvi = base_ndvi + trend + seasonal + noise
                ndvi = max(0.1, min(0.9, ndvi))
                
                data.append({
                    'palm_id': palm_id,
                    'date': date,
                    'ndvi': ndvi,
                    'gndvi': ndvi * 0.9 + np.random.normal(0, 0.01),
                    'ndre': ndvi * 0.8 + np.random.normal(0, 0.01)
                })
        
        df = pd.DataFrame(data)
        st.session_state.state.time_series_data = df
    
    df = st.session_state.state.time_series_data
    
    # Selector de palma para análisis
    palm_ids = df['palm_id'].unique()
    selected_palm = st.selectbox("Seleccionar Palma para Análisis", palm_ids)
    
    # Filtrar datos
    palm_data = df[df['palm_id'] == selected_palm]
    
    # Gráfico de series temporales
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=palm_data['date'], y=palm_data['ndvi'],
                            mode='lines+markers', name='NDVI',
                            line=dict(color='green', width=3)))
    
    # Líneas de referencia para estrés
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                  annotation_text="Umbral Estrés Leve")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                  annotation_text="Umbral Estrés Severo")
    
    fig.update_layout(
        title=f"Evolución de NDVI - {selected_palm}",
        xaxis_title="Fecha",
        yaxis_title="NDVI",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de tendencia
    st.subheader("Análisis de Tendencia")
    
    # Calcular tendencia
    from scipy import stats
    x = np.arange(len(palm_data))
    y = palm_data['ndvi'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        trend_icon = "📈" if slope > 0 else "📉"
        st.metric("Tendencia", f"{slope:.4f} por semana", delta=trend_icon)
    with col2:
        st.metric("NDVI Actual", f"{y[-1]:.3f}")
    with col3:
        status = "Óptimo" if y[-1] > 0.7 else "Saludable" if y[-1] > 0.6 else "En Riesgo"
        st.metric("Estado Actual", status)

# Módulo 5: Alertas Tempranas
elif app_mode == "⚠️ Alertas Tempranas":
    st.header("Sistema de Alertas Tempranas")
    
    # Simular alertas
    alerts = [
        {
            'palm_id': 'P023',
            'type': 'STRESS',
            'severity': 'MEDIUM',
            'message': 'NDVI descendió de 0.68 a 0.58 en 2 semanas',
            'date': '2024-03-20',
            'confidence': 0.87
        },
        {
            'palm_id': 'P045',
            'type': 'COMPETITION',
            'severity': 'LOW',
            'message': 'Distancia con vecina: 6.2m (óptimo: 7.5m)',
            'date': '2024-03-19',
            'confidence': 0.72
        },
        {
            'palm_id': 'P118',
            'type': 'DISEASE_RISK',
            'severity': 'HIGH',
            'message': 'Patrón anómalo detectado - posible infección temprana',
            'date': '2024-03-18',
            'confidence': 0.91
        }
    ]
    
    for alert in alerts:
        if alert['severity'] == 'HIGH':
            color = "red"
            icon = "🔴"
        elif alert['severity'] == 'MEDIUM':
            color = "orange"
            icon = "🟠"
        else:
            color = "yellow"
            icon = "🟡"
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding: 10px; background-color: #f8f9fa; margin: 10px 0;">
                <h4>{icon} Alerta {alert['type']} - {alert['palm_id']}</h4>
                <p><strong>Severidad:</strong> {alert['severity']} | <strong>Confianza:</strong> {alert['confidence']*100}%</p>
                <p>{alert['message']}</p>
                <small>Detectado: {alert['date']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Ver detalles {alert['palm_id']}"):
                    st.session_state.selected_palm = alert['palm_id']
                    st.experimental_rerun()
            with col2:
                if st.button(f"Marcar como revisada {alert['palm_id']}"):
                    st.success(f"Alerta {alert['palm_id']} marcada como revisada")
    
    # Modelo predictivo
    st.subheader("Predicción de Estrés")
    
    if st.button("Ejecutar Modelo Predictivo"):
        with st.spinner("Analizando patrones temporales..."):
            # Simular predicción
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            st.success("Análisis completado")
            
            # Mostrar resultados de predicción
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Riesgo en Próximas 4 Semanas")
                risk_data = {
                    'Bajo Riesgo': 72,
                    'Riesgo Moderado': 18,
                    'Alto Riesgo': 8,
                    'Riesgo Crítico': 2
                }
                fig = px.pie(values=list(risk_data.values()), 
                           names=list(risk_data.keys()))
                st.plotly_chart(fig, use_container_width=True)

# Módulo 6: Mapa de Distribución
elif app_mode == "🗺️ Mapa de Distribución":
    st.header("Mapa de Distribución y Densidad")
    
    # Crear datos de ejemplo para el mapa
    lons = np.random.uniform(-74.2, -74.0, 100)
    lats = np.random.uniform(4.5, 4.7, 100)
    ndvi_values = np.random.uniform(0.3, 0.85, 100)
    status = ['Óptimo' if x > 0.7 else 'Saludable' if x > 0.6 else 'Estrés' for x in ndvi_values]
    
    map_data = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'ndvi': ndvi_values,
        'status': status,
        'palm_id': [f'P{i:03d}' for i in range(1, 101)]
    })
    
    # Crear mapa interactivo
    fig = px.scatter_mapbox(map_data, 
                          lat="lat", 
                          lon="lon", 
                          color="status",
                          size="ndvi",
                          hover_name="palm_id",
                          hover_data={"ndvi": True, "status": True},
                          color_discrete_map={
                              'Óptimo': 'green',
                              'Saludable': 'lightgreen', 
                              'Estrés': 'red'
                          },
                          zoom=10,
                          height=600)
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Controles del mapa
    col1, col2 = st.columns(2)
    with col1:
        show_heatmap = st.checkbox("Mostrar mapa de calor de NDVI")
        show_voronoi = st.checkbox("Mostrar diagrama de Voronoi")
    
    with col2:
        selected_status = st.multiselect(
            "Filtrar por estado",
            ['Óptimo', 'Saludable', 'Estrés'],
            default=['Óptimo', 'Saludable', 'Estrés']
        )

# Footer
st.markdown("---")
st.markdown(
    "🌴 **Sistema de Monitoreo Inteligente - Palma Aceitera** | "
    "Desarrollado con Streamlit & Python | "
    "© 2024"
)

if __name__ == "__main__":
    pass
