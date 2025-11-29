import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

# Función de detección SIMULADA (sin OpenCV)
def simulate_palm_detection(image_size=(800, 600), num_palms=25):
    """Simula la detección de palmas sin OpenCV"""
    try:
        palm_data = []
        for i in range(num_palms):
            palm_data.append({
                'id': f'P{i+1:03d}',
                'x': np.random.randint(50, image_size[0]-50),
                'y': np.random.randint(50, image_size[1]-50),
                'area': np.random.randint(80, 200)
            })
        return palm_data
    except Exception as e:
        st.error(f"Error en simulación: {e}")
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
        st.metric("Palmas Monitoreadas", "1,247", "+12", key="metric1")
    with col2:
        st.metric("Tasa de Estrés", "8.2%", "-2.1%", key="metric2")
    with col3:
        st.metric("Distancia Promedio", "7.8m", "Óptima", key="metric3")
    with col4:
        st.metric("NDVI Promedio", "0.74", "+0.03", key="metric4")
    
    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Salud")
        health_data = pd.DataFrame({
            'Estado': ['Óptimo', 'Saludable', 'Estrés Leve', 'Estrés Severo'],
            'Cantidad': [650, 450, 120, 27]
        })
        fig = px.pie(health_data, values='Cantidad', names='Estado')
        st.plotly_chart(fig, use_container_width=True, key="pie1")
    
    with col2:
        st.subheader("Tendencia NDVI")
        dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
        ndvi_trend = [0.72, 0.73, 0.74, 0.72, 0.71, 0.70, 0.69, 0.68, 0.70, 0.72, 0.73, 0.74]
        fig = px.line(x=dates, y=ndvi_trend, title="NDVI Semanal")
        fig.update_layout(xaxis_title="Fecha", yaxis_title="NDVI")
        st.plotly_chart(fig, use_container_width=True, key="line1")

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
        st.info("💡 **Demo**: Puedes usar cualquier imagen para probar la funcionalidad")
    else:
        col1, col2 = st.columns(2)
        with col1:
            num_palms = st.slider("Número de palmas a simular", 5, 50, 25)
        
        with col2:
            if st.button("Simular Detección"):
                with st.spinner("Simulando detección de palmas..."):
                    try:
                        # Usar simulación en lugar de OpenCV
                        palm_data = simulate_palm_detection(num_palms=num_palms)
                        st.session_state.state.palm_data = palm_data
                        st.success(f"✅ {len(palm_data)} palmas detectadas (simulación)")
                    except Exception as e:
                        st.error(f"Error en simulación: {e}")
        
        if st.session_state.state.palm_data:
            st.subheader("Resultados de Detección")
            df_palms = pd.DataFrame(st.session_state.state.palm_data)
            
            # Mostrar tabla
            st.dataframe(df_palms, use_container_width=True)
            
            # Gráfico de dispersión
            fig = px.scatter(df_palms, x='x', y='y', size='area', 
                           title="Distribución de Palmeras Detectadas (Simulación)", 
                           hover_data=['id'], color='area')
            st.plotly_chart(fig, use_container_width=True, key="scatter1")

elif st.session_state.state.current_page == "📈 Análisis":
    st.header("Análisis de Series Temporales")
    
    # Generar datos de ejemplo
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
                'ndvi': ndvi,
                'gndvi': ndvi * 0.9 + np.random.normal(0, 0.01),
                'ndre': ndvi * 0.8 + np.random.normal(0, 0.01)
            })
    
    df = pd.DataFrame(data)
    
    # Selector de palma
    selected_palm = st.selectbox("Seleccionar Palma para Análisis", palm_ids, key="palm_selector")
    palm_data = df[df['palm_id'] == selected_palm]
    
    # Gráfico de series temporales
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=palm_data['date'], y=palm_data['ndvi'],
                            mode='lines+markers', name='NDVI',
                            line=dict(color='green', width=3)))
    
    # Líneas de referencia
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="Óptimo", annotation_position="right")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                  annotation_text="Estrés Leve", annotation_position="right")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                  annotation_text="Estrés Severo", annotation_position="right")
    
    fig.update_layout(
        title=f"Evolución de NDVI - {selected_palm}",
        xaxis_title="Fecha",
        yaxis_title="NDVI",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key="ndvi_chart")
    
    # Análisis de tendencia
    st.subheader("Análisis de Tendencia")
    current_ndvi = palm_data['ndvi'].iloc[-1]
    trend = (palm_data['ndvi'].iloc[-1] - palm_data['ndvi'].iloc[0]) / len(palm_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NDVI Actual", f"{current_ndvi:.3f}")
    with col2:
        trend_icon = "📈" if trend > 0 else "📉"
        st.metric("Tendencia", f"{trend:.4f}/semana", delta=trend_icon)
    with col3:
        if current_ndvi > 0.7:
            status = "🟢 Óptimo"
        elif current_ndvi > 0.6:
            status = "🟡 Saludable"
        elif current_ndvi > 0.4:
            status = "🟠 Estrés Leve"
        else:
            status = "🔴 Estrés Severo"
        st.metric("Estado", status)

elif st.session_state.state.current_page == "⚠️ Alertas":
    st.header("Sistema de Alertas Tempranas")
    
    # Alertas de ejemplo
    alertas = [
        {'id': 'P023', 'tipo': 'Estrés', 'severidad': 'Media', 
         'mensaje': 'NDVI descendió de 0.68 a 0.58 en 2 semanas', 'fecha': '2024-03-20'},
        {'id': 'P045', 'tipo': 'Competencia', 'severidad': 'Baja', 
         'mensaje': 'Distancia con vecina: 6.2m (óptimo: 7.5m)', 'fecha': '2024-03-19'},
        {'id': 'P118', 'tipo': 'Riesgo Enfermedad', 'severidad': 'Alta', 
         'mensaje': 'Patrón anómalo detectado - posible infección temprana', 'fecha': '2024-03-18'}
    ]
    
    for alerta in alertas:
        if alerta['severidad'] == 'Alta':
            st.markdown(f"""
            <div class="alert-high">
                <h4>🔴 {alerta['tipo']} - {alerta['id']}</h4>
                <p>{alerta['mensaje']}</p>
                <small>Detectado: {alerta['fecha']}</small>
            </div>
            """, unsafe_allow_html=True)
        elif alerta['severidad'] == 'Media':
            st.markdown(f"""
            <div class="alert-medium">
                <h4>🟠 {alerta['tipo']} - {alerta['id']}</h4>
                <p>{alerta['mensaje']}</p>
                <small>Detectado: {alerta['fecha']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                <h4>🟡 {alerta['tipo']} - {alerta['id']}</h4>
                <p>{alerta['mensaje']}</p>
                <small>Detectado: {alerta['fecha']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Botón para generar reporte
    if st.button("Generar Reporte de Alertas"):
        st.download_button(
            label="📥 Descargar Reporte CSV",
            data=pd.DataFrame(alertas).to_csv(index=False),
            file_name="alertas_palmas.csv",
            mime="text/csv"
        )

elif st.session_state.state.current_page == "🗺️ Mapa":
    st.header("Mapa de Distribución y Densidad")
    
    # Generar datos de ejemplo para el mapa
    np.random.seed(42)  # Para reproducibilidad
    lons = np.random.uniform(-74.2, -74.0, 80)
    lats = np.random.uniform(4.5, 4.7, 80)
    ndvi_values = np.random.uniform(0.3, 0.85, 80)
    
    # Clasificar por estado de salud
    status = []
    for ndvi in ndvi_values:
        if ndvi > 0.7:
            status.append('Óptimo')
        elif ndvi > 0.6:
            status.append('Saludable')
        elif ndvi > 0.4:
            status.append('Estrés Leve')
        else:
            status.append('Estrés Severo')
    
    map_data = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'ndvi': ndvi_values,
        'status': status,
        'palm_id': [f'P{i:03d}' for i in range(1, 81)]
    })
    
    # Mapa interactivo
    fig = px.scatter_mapbox(map_data, 
                          lat="lat", 
                          lon="lon", 
                          color="status",
                          size="ndvi",
                          hover_name="palm_id",
                          hover_data={"ndvi": True, "status": True, "lat": False, "lon": False},
                          color_discrete_map={
                              'Óptimo': 'green',
                              'Saludable': 'lightgreen', 
                              'Estrés Leve': 'orange',
                              'Estrés Severo': 'red'
                          },
                          zoom=10,
                          height=600)
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    st.plotly_chart(fig, use_container_width=True, key="map1")
    
    # Estadísticas del mapa
    st.subheader("Estadísticas de Distribución")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Palmeras", len(map_data))
    with col2:
        healthy_count = len(map_data[map_data['status'].isin(['Óptimo', 'Saludable'])])
        st.metric("Palmeras Saludables", f"{healthy_count} ({healthy_count/len(map_data)*100:.1f}%)")
    with col3:
        avg_ndvi = map_data['ndvi'].mean()
        st.metric("NDVI Promedio", f"{avg_ndvi:.3f}")

# Footer
st.markdown("---")
st.markdown("🌴 **Sistema de Monitoreo Inteligente - Palma Aceitera** | © 2024")
