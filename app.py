import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from PIL import Image
import json
import io
import time
import base64

# Configuración estable
st.set_page_config(
    page_title="Monitoreo Inteligente - Palma Aceitera",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS mejorado
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2e7d32 !important; border-bottom: 3px solid #4caf50; padding-bottom: 10px; }
    h2 { color: #388e3c !important; margin-top: 20px !important; }
    [data-testid="metric-container"] { 
        background-color: white; border-radius: 10px; padding: 15px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .stButton>button { 
        background-color: #4caf50; color: white; border-radius: 5px; 
        border: none; padding: 10px 20px; font-weight: bold; 
    }
    .alert-high { border-left: 5px solid #d32f2f; background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .alert-medium { border-left: 5px solid #ff9800; background-color: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .alert-low { border-left: 5px solid #ffeb3b; background-color: #fffde7; padding: 15px; margin: 10px 0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Clase de estado mejorada
class SessionState:
    def __init__(self):
        self.uploaded_images = []
        self.palm_data = []
        self.time_series_data = None
        self.distribution_analysis = None
        self.current_page = "Dashboard"

# Inicializar estado
if 'state' not in st.session_state:
    st.session_state.state = SessionState()

# Título principal
st.title("🌴 Sistema de Monitoreo Inteligente - Palma Aceitera")
st.markdown("---")

# Sidebar simplificado
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/4CAF50/FFFFFF?text=PalmaApp", width=150)
    st.title("Navegación")
    
    # Navegación por radio buttons (más estable)
    page = st.radio(
        "Selecciona el módulo:",
        ["📊 Dashboard", "🛰️ Cargar Datos", "🔍 Detección", "📈 Análisis", "⚠️ Alertas", "🗺️ Mapa"],
        key="nav_radio"
    )
    
    st.session_state.state.current_page = page

# Funciones de utilidad
def segment_individual_palms(image_array, min_area=50):
    """Detección simplificada de palmas usando OpenCV"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        palm_data = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    palm_data.append({
                        'id': f'P{i+1:03d}', 'x': cx, 'y': cy, 'area': area
                    })
        
        return palm_data
    except Exception as e:
        st.error(f"Error en detección: {e}")
        # Datos de ejemplo como fallback
        return [
            {'id': 'P001', 'x': 100, 'y': 100, 'area': 150},
            {'id': 'P002', 'x': 200, 'y': 150, 'area': 120},
            {'id': 'P003', 'x': 300, 'y': 200, 'area': 180}
        ]

# Contenido principal según página seleccionada
if st.session_state.state.current_page == "📊 Dashboard":
    st.header("Dashboard de Monitoreo")
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Palmas Monitoreadas", "1,247", "+12")
    with col2:
        st.metric("Tasa de Estrés", "8.2%", "-2.1%")
    with col3:
        st.metric("Distancia Promedio", "7.8m", "Óptima")
    with col4:
        st.metric("NDVI Promedio", "0.74", "+0.03")
    
    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Salud")
        health_data = pd.DataFrame({
            'Estado': ['Óptimo', 'Saludable', 'Estrés Leve', 'Estrés Severo'],
            'Cantidad': [650, 450, 120, 27]
        })
        fig = px.pie(health_data, values='Cantidad', names='Estado')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tendencia NDVI")
        dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
        ndvi_trend = [0.72, 0.73, 0.74, 0.72, 0.71, 0.70, 0.69, 0.68, 0.70, 0.72, 0.73, 0.74]
        fig = px.line(x=dates, y=ndvi_trend, title="NDVI Semanal")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="NDVI")
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.state.current_page == "🛰️ Cargar Datos":
    st.header("Carga de Datos Multiespectrales")
    
    uploaded_files = st.file_uploader(
        "Carga imágenes (PNG, JPG)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.session_state.state.uploaded_images = uploaded_files
        st.success(f"{len(uploaded_files)} imágenes cargadas")
        
        # Preview
        st.subheader("Vista Previa")
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files[:3]):
            with cols[idx]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Imagen {idx+1}", use_column_width=True)

elif st.session_state.state.current_page == "🔍 Detección":
    st.header("Detección Individual de Palmeras")
    
    if not st.session_state.state.uploaded_images:
        st.warning("Primero carga imágenes en 'Cargar Datos'")
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_area = st.slider("Área mínima", 10, 100, 50)
        
        with col2:
            if st.button("Iniciar Detección"):
                with st.spinner("Procesando..."):
                    try:
                        uploaded_file = st.session_state.state.uploaded_images[0]
                        image = Image.open(uploaded_file)
                        image_array = np.array(image)
                        
                        palm_data = segment_individual_palms(image_array, min_area)
                        st.session_state.state.palm_data = palm_data
                        st.success(f"✅ {len(palm_data)} palmas detectadas")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.session_state.state.palm_data:
            st.subheader("Resultados")
            df_palms = pd.DataFrame(st.session_state.state.palm_data)
            fig = px.scatter(df_palms, x='x', y='y', size='area', 
                           title="Palmeras Detectadas", hover_data=['id'])
            st.plotly_chart(fig, use_container_width=True)

elif st.session_state.state.current_page == "📈 Análisis":
    st.header("Análisis de Series Temporales")
    
    # Datos de ejemplo
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='W')
    palm_ids = [f'P{i:03d}' for i in range(1, 6)]
    
    data = []
    for palm_id in palm_ids:
        base_ndvi = np.random.normal(0.75, 0.05)
        for date in dates:
            trend = 0.01 * (date - dates[0]).days / 7
            noise = np.random.normal(0, 0.02)
            seasonal = 0.03 * np.sin(2 * np.pi * (date - dates[0]).days / 30)
            ndvi = max(0.1, min(0.9, base_ndvi + trend + seasonal + noise))
            
            data.append({
                'palm_id': palm_id,
                'date': date,
                'ndvi': ndvi
            })
    
    df = pd.DataFrame(data)
    
    selected_palm = st.selectbox("Seleccionar Palma", palm_ids)
    palm_data = df[df['palm_id'] == selected_palm]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=palm_data['date'], y=palm_data['ndvi'],
                            mode='lines+markers', name='NDVI',
                            line=dict(color='green', width=3)))
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red")
    
    fig.update_layout(title=f"NDVI - {selected_palm}", height=400)
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.state.current_page == "⚠️ Alertas":
    st.header("Sistema de Alertas Tempranas")
    
    alertas = [
        {'id': 'P023', 'tipo': 'Estrés', 'severidad': 'Media', 'mensaje': 'NDVI descendió 0.68→0.58'},
        {'id': 'P045', 'tipo': 'Competencia', 'severidad': 'Baja', 'mensaje': 'Distancia: 6.2m (óptimo: 7.5m)'},
        {'id': 'P118', 'tipo': 'Riesgo Enfermedad', 'severidad': 'Alta', 'mensaje': 'Patrón anómalo detectado'}
    ]
    
    for alerta in alertas:
        if alerta['severidad'] == 'Alta':
            st.markdown(f"""
            <div class="alert-high">
                <h4>🔴 {alerta['tipo']} - {alerta['id']}</h4>
                <p>{alerta['mensaje']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif alerta['severidad'] == 'Media':
            st.markdown(f"""
            <div class="alert-medium">
                <h4>🟠 {alerta['tipo']} - {alerta['id']}</h4>
                <p>{alerta['mensaje']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                <h4>🟡 {alerta['tipo']} - {alerta['id']}</h4>
                <p>{alerta['mensaje']}</p>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.state.current_page == "🗺️ Mapa":
    st.header("Mapa de Distribución")
    
    # Datos de ejemplo para el mapa
    lons = np.random.uniform(-74.2, -74.0, 50)
    lats = np.random.uniform(4.5, 4.7, 50)
    ndvi_values = np.random.uniform(0.3, 0.85, 50)
    
    map_data = pd.DataFrame({
        'lat': lats, 'lon': lons, 'ndvi': ndvi_values,
        'palm_id': [f'P{i:03d}' for i in range(1, 51)]
    })
    
    fig = px.scatter_mapbox(map_data, lat="lat", lon="lon", color="ndvi",
                          size="ndvi", hover_name="palm_id", zoom=10, height=600,
                          color_continuous_scale=["red", "yellow", "green"])
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🌴 **Sistema de Monitoreo Inteligente** | © 2024")
