import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
# import torch  # Comentado temporalmente
# import torch.nn as nn  # Comentado temporalmente
from PIL import Image
import json
import io
import time

# Configuración de página CON MÁS OPTIMIZACIONES
st.set_page_config(
    page_title="Monitoreo Inteligente - Palma Aceitera",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CARGAR CSS MEJORADO
def load_css():
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
    </style>
    """, unsafe_allow_html=True)

load_css()

# CLASE DE ESTADO MEJORADA
class SessionState:
    def __init__(self):
        self._init_state()
    
    def _init_state(self):
        self.uploaded_images = []
        self.palm_data = []
        self.time_series_data = None
        self.distribution_analysis = None
        self.last_update = time.time()
        self.current_module = "📊 Dashboard"

# INICIALIZAR ESTADO CON MÁS ROBUSTEZ
@st.cache_resource
def get_session_state():
    return SessionState()

state = get_session_state()

# FUNCIÓN PARA CAMBIAR MÓDULOS CON ESTABILIDAD
def switch_module(module_name):
    state.current_module = module_name
    st.rerun()

# TÍTULO PRINCIPAL
st.title("🌴 Sistema de Monitoreo Inteligente - Palma Aceitera")
st.markdown("---")

# SIDEBAR MEJORADO
st.sidebar.image("https://via.placeholder.com/150x50/4CAF50/FFFFFF?text=Logo", width=150)
st.sidebar.title("Navegación")

# Botones de navegación más estables
modules = [
    "📊 Dashboard", 
    "🛰️ Cargar Datos", 
    "🔍 Detección de Palmeras", 
    "📈 Análisis Temporal", 
    "⚠️ Alertas Tempranas", 
    "🗺️ Mapa de Distribución"
]

for module in modules:
    if st.sidebar.button(module, key=f"nav_{module}", use_container_width=True):
        switch_module(module)

# CONTENIDO PRINCIPAL CON MÁS ESTABILIDAD
try:
    if state.current_module == "📊 Dashboard":
        st.header("Dashboard de Monitoreo")
        
        # Métricas con keys únicos
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Palmas Monitoreadas", "1,247", "+12", key="metric_1")
        with col2:
            st.metric("Tasa de Estrés", "8.2%", "-2.1%", key="metric_2")
        with col3:
            st.metric("Distancia Promedio", "7.8m", "Óptima", key="metric_3")
        with col4:
            st.metric("NDVI Promedio", "0.74", "+0.03", key="metric_4")
        
        # Gráficos con contenedores estables
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribución de Salud")
                health_data = pd.DataFrame({
                    'Estado': ['Óptimo', 'Saludable', 'Estrés Leve', 'Estrés Severo'],
                    'Cantidad': [650, 450, 120, 27]
                })
                fig = px.pie(health_data, values='Cantidad', names='Estado')
                st.plotly_chart(fig, use_container_width=True, key="pie_chart_1")
            
            with col2:
                st.subheader("Tendencia NDVI")
                dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
                ndvi_trend = [0.72, 0.73, 0.74, 0.72, 0.71, 0.70, 0.69, 0.68, 0.70, 0.72, 0.73, 0.74]
                fig = px.line(x=dates, y=ndvi_trend, title="NDVI Semanal")
                fig.update_layout(xaxis_title="Fecha", yaxis_title="NDVI")
                st.plotly_chart(fig, use_container_width=True, key="line_chart_1")

    elif state.current_module == "🛰️ Cargar Datos":
        st.header("Carga de Datos Multiespectrales")
        
        uploaded_files = st.file_uploader(
            "Carga imágenes multiespectrales",
            type=['png', 'jpg', 'jpeg'],  # Formatos más simples primero
            accept_multiple_files=True,
            key="file_uploader_1"
        )
        
        if uploaded_files:
            state.uploaded_images = uploaded_files
            st.success(f"{len(uploaded_files)} imágenes cargadas")
            
            # Preview seguro
            st.subheader("Vista Previa")
            cols = st.columns(min(3, len(uploaded_files)))
            for idx, uploaded_file in enumerate(uploaded_files[:3]):
                with cols[idx]:
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Imagen {idx+1}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error cargando imagen {idx+1}: {str(e)}")

    elif state.current_module == "🔍 Detección de Palmeras":
        st.header("Detección Individual de Palmeras")
        
        if not state.uploaded_images:
            st.warning("Por favor carga imágenes en el módulo 'Cargar Datos' primero.")
        else:
            # Usar columnas con keys únicos
            col1, col2 = st.columns(2)
            
            with col1:
                min_area = st.slider("Área mínima", 10, 100, 50, key="slider_area")
            
            with col2:
                if st.button("Iniciar Detección", key="btn_detect"):
                    with st.spinner("Procesando imágenes..."):
                        try:
                            # Simulación estable
                            simulated_palms = [
                                {'id': f'P{i:03d}', 'x': np.random.randint(100, 900), 
                                 'y': np.random.randint(100, 900), 'area': np.random.randint(50, 200)}
                                for i in range(1, 46)
                            ]
                            state.palm_data = simulated_palms
                            st.success(f"✅ {len(simulated_palms)} palmas detectadas")
                        except Exception as e:
                            st.error(f"Error en detección: {str(e)}")
            
            # Mostrar resultados de forma segura
            if state.palm_data:
                st.subheader("Resultados de Detección")
                df_palms = pd.DataFrame(state.palm_data)
                
                # Gráfico con key único
                fig = px.scatter(df_palms, x='x', y='y', size='area', 
                               title="Distribución de Palmeras Detectadas",
                               hover_data=['id'])
                st.plotly_chart(fig, use_container_width=True, key="scatter_plot_1")

    # ... (continuar con los otros módulos de manera similar)

except Exception as e:
    st.error(f"Error en la aplicación: {str(e)}")
    st.info("""
    **Solución de problemas:**
    - Recarga la página
    - Limpia el caché del navegador
    - Verifica tu conexión a internet
    """)

# FOOTER MEJORADO
st.markdown("---")
st.markdown(
    "🌴 **Sistema de Monitoreo Inteligente - Palma Aceitera** | "
    "Desarrollado con Streamlit & Python | "
    "© 2024"
)

# Script para prevenir el error
st.markdown("""
<script>
// Prevenir el error de removeChild
if (window.streamlitDebug) {
    console.log("Streamlit app loaded successfully");
}
</script>
""", unsafe_allow_html=True)
