import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuración básica y estable
st.set_page_config(
    page_title="Monitoreo Palma Aceitera",
    page_icon="🌴",
    layout="wide"
)

# CSS mínimo
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2e7d32; border-bottom: 2px solid #4caf50; padding-bottom: 10px; }
    .metric-box { 
        background: white; padding: 15px; border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Estado simple
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'palm_data' not in st.session_state:
    st.session_state.palm_data = []

# Título
st.title("🌴 Monitoreo Inteligente - Palma Aceitera")
st.markdown("---")

# Sidebar simple
with st.sidebar:
    st.title("Navegación")
    page = st.radio("Módulos:", [
        "📊 Dashboard", 
        "🛰️ Cargar Datos", 
        "🔍 Detección", 
        "📈 Análisis", 
        "⚠️ Alertas"
    ])
    st.session_state.page = page

# Simulación de detección
def simulate_detection(num_palms=20):
    return [
        {'id': f'P{i:03d}', 'x': np.random.randint(50, 750), 
         'y': np.random.randint(50, 550), 'area': np.random.randint(50, 200)}
        for i in range(1, num_palms + 1)
    ]

# Contenido según página
if st.session_state.page == "📊 Dashboard":
    st.header("Dashboard Principal")
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Palmas Monitoreadas", "1,247", "+12")
    with col2:
        st.metric("Tasa de Estrés", "8.2%", "-2.1%")
    with col3:
        st.metric("Distancia Promedio", "7.8m")
    with col4:
        st.metric("NDVI Promedio", "0.74", "+0.03")
    
    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Salud por Lote")
        data = pd.DataFrame({
            'Estado': ['Óptimo', 'Saludable', 'Estrés Leve', 'Estrés Severo'],
            'Cantidad': [650, 450, 120, 27]
        })
        fig = px.pie(data, values='Cantidad', names='Estado')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tendencia NDVI")
        dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
        ndvi = [0.72, 0.73, 0.74, 0.72, 0.71, 0.70, 0.69, 0.68, 0.70, 0.72, 0.73, 0.74]
        fig = px.line(x=dates, y=ndvi, labels={'x': 'Fecha', 'y': 'NDVI'})
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "🛰️ Cargar Datos":
    st.header("Carga de Imágenes")
    
    uploaded_files = st.file_uploader(
        "Sube imágenes de palmas (PNG, JPG)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"{len(uploaded_files)} imágenes cargadas")
        
        # Mostrar preview
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files[:3]):
            with cols[idx]:
                image = Image.open(file)
                st.image(image, use_column_width=True)

elif st.session_state.page == "🔍 Detección":
    st.header("Detección de Palmeras")
    
    if not st.session_state.uploaded_images:
        st.warning("Primero carga imágenes en 'Cargar Datos'")
    else:
        if st.button("Ejecutar Detección"):
            with st.spinner("Detectando palmas..."):
                st.session_state.palm_data = simulate_detection(25)
                st.success(f"✅ {len(st.session_state.palm_data)} palmas detectadas")
        
        if st.session_state.palm_data:
            df = pd.DataFrame(st.session_state.palm_data)
            st.subheader(f"Resultados: {len(df)} palmas")
            
            fig = px.scatter(df, x='x', y='y', size='area', 
                           color='area', hover_data=['id'])
            st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "📈 Análisis":
    st.header("Análisis Temporal")
    
    # Datos de ejemplo
    dates = pd.date_range(start='2024-01-01', periods=8, freq='W')
    palms = [f'P{i:03d}' for i in range(1, 6)]
    
    data = []
    for palm in palms:
        base = np.random.normal(0.75, 0.05)
        for date in dates:
            noise = np.random.normal(0, 0.02)
            ndvi = max(0.3, min(0.9, base + noise))
            data.append({'palm_id': palm, 'date': date, 'ndvi': ndvi})
    
    df = pd.DataFrame(data)
    
    selected_palm = st.selectbox("Seleccionar Palma", palms)
    palm_df = df[df['palm_id'] == selected_palm]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=palm_df['date'], y=palm_df['ndvi'],
                            mode='lines+markers', name='NDVI',
                            line=dict(color='green', width=3)))
    
    # Líneas de referencia
    fig.add_hline(y=0.7, line_dash="dash", line_color="green")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange") 
    fig.add_hline(y=0.4, line_dash="dash", line_color="red")
    
    fig.update_layout(title=f"NDVI - {selected_palm}", height=400)
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "⚠️ Alertas":
    st.header("Alertas del Sistema")
    
    alertas = [
        {"tipo": "Estrés", "palma": "P023", "mensaje": "NDVI bajo detectado", "severidad": "Media"},
        {"tipo": "Competencia", "palma": "P045", "mensaje": "Distancia muy cercana", "severidad": "Baja"},
        {"tipo": "Riesgo", "palma": "P118", "mensaje": "Posible enfermedad", "severidad": "Alta"}
    ]
    
    for alerta in alertas:
        if alerta["severidad"] == "Alta":
            st.error(f"🔴 {alerta['tipo']} - {alerta['palma']}: {alerta['mensaje']}")
        elif alerta["severidad"] == "Media":
            st.warning(f"🟠 {alerta['tipo']} - {alerta['palma']}: {alerta['mensaje']}")
        else:
            st.info(f"🟡 {alerta['tipo']} - {alerta['palma']}: {alerta['mensaje']}")

# Footer
st.markdown("---")
st.markdown("🌴 *Sistema de Monitoreo - Versión Estable*")
