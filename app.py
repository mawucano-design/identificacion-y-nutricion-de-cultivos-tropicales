import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from pykml import parser
from io import BytesIO
import xml.etree.ElementTree as ET

# Configuración
st.set_page_config(
    page_title="Monitoreo Palma Aceitera", 
    page_icon="🌴", 
    layout="wide"
)

# CSS mejorado
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2e7d32; border-bottom: 2px solid #4caf50; padding-bottom: 10px; }
    .success-box { background: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; }
    .info-box { background: #d1ecf1; padding: 15px; border-radius: 5px; border-left: 5px solid #17a2b8; }
</style>
""", unsafe_allow_html=True)

# Estado de la sesión
if 'kml_data' not in st.session_state:
    st.session_state.kml_data = None
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'palm_data' not in st.session_state:
    st.session_state.palm_data = []

# Título
st.title("🌴 Monitoreo Inteligente - Palma Aceitera")
st.markdown("---")

# Navegación
page = st.sidebar.radio("Navegación", [
    "📊 Dashboard", 
    "🗺️ Cargar KML", 
    "🛰️ Cargar Imágenes", 
    "🔍 Detección", 
    "📈 Análisis"
])

# Función para procesar KML
def process_kml(uploaded_file):
    """Procesa archivo KML y extrae polígonos"""
    try:
        # Leer el contenido del archivo
        kml_content = uploaded_file.read()
        
        # Intentar parsear con geopandas
        gdf = gpd.read_file(BytesIO(kml_content))
        
        # Extraer coordenadas del primer polígono
        if not gdf.empty and hasattr(gdf.geometry.iloc[0], 'exterior'):
            polygon = gdf.geometry.iloc[0]
            coords = list(polygon.exterior.coords)
            
            # Convertir a formato para Folium
            polygon_coords = [[lat, lon] for lon, lat in coords]
            
            return {
                'gdf': gdf,
                'polygon_coords': polygon_coords,
                'centroid': [polygon.centroid.y, polygon.centroid.x],
                'area_ha': polygon.area * 10000  # Convertir a hectáreas
            }
        else:
            st.error("No se encontraron polígonos válidos en el archivo KML")
            return None
            
    except Exception as e:
        st.error(f"Error procesando KML: {str(e)}")
        return None

# Función para crear mapa con polígono
def create_folium_map(polygon_data):
    """Crea un mapa Folium con el polígono KML"""
    if not polygon_data:
        return None
        
    center = polygon_data['centroid']
    m = folium.Map(location=center, zoom_start=14)
    
    # Agregar polígono
    folium.Polygon(
        locations=polygon_data['polygon_coords'],
        popup=f"Área: {polygon_data['area_ha']:.2f} ha",
        color='#4CAF50',
        fill=True,
        fillColor='#4CAF50',
        fillOpacity=0.3
    ).add_to(m)
    
    # Agregar marcador en el centroide
    folium.Marker(
        center,
        popup="Centro del lote",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m

# Páginas
if page == "📊 Dashboard":
    st.header("Dashboard Principal")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lotes Cargados", "3", "1 nuevo")
    with col2:
        st.metric("Área Total", "45.2 ha")
    with col3:
        st.metric("Palmas Detectadas", "1,247")
    with col4:
        st.metric("NDVI Promedio", "0.74", "+0.03")
    
    # Mapa de ejemplo
    st.subheader("Ubicación de Lotes")
    if st.session_state.kml_data:
        folium_map = create_folium_map(st.session_state.kml_data)
        if folium_map:
            folium_static(folium_map, width=800, height=400)
    else:
        # Mapa por defecto
        m = folium.Map(location=[4.6, -74.1], zoom_start=6)
        folium_static(m, width=800, height=400)
        st.info("💡 Carga un archivo KML en la pestaña 'Cargar KML' para ver tu lote")

elif page == "🗺️ Cargar KML":
    st.header("Cargar Polígono desde KML")
    
    st.info("""
    **Formatos soportados:** KML, KMZ
    **Recomendación:** Exporta tu polígono desde Google Earth, QGIS o ArcGIS
    """)
    
    uploaded_kml = st.file_uploader(
        "Selecciona archivo KML", 
        type=['kml', 'kmz'],
        key="kml_uploader"
    )
    
    if uploaded_kml:
        with st.spinner("Procesando polígono KML..."):
            polygon_data = process_kml(uploaded_kml)
            
            if polygon_data:
                st.session_state.kml_data = polygon_data
                
                st.success(f"✅ Polígono cargado exitosamente")
                
                # Mostrar información del polígono
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Área del Lote", f"{polygon_data['area_ha']:.2f} ha")
                with col2:
                    centroid = polygon_data['centroid']
                    st.metric("Centroide", f"{centroid[0]:.4f}, {centroid[1]:.4f}")
                with col3:
                    st.metric("Vértices", f"{len(polygon_data['polygon_coords'])}")
                
                # Mostrar mapa
                st.subheader("Visualización del Polígono")
                folium_map = create_folium_map(polygon_data)
                if folium_map:
                    folium_static(folium_map, width=800, height=500)
                
                # Mostrar datos técnicos
                with st.expander("📊 Datos Técnicos del Polígono"):
                    st.write(polygon_data['gdf'])

elif page == "🛰️ Cargar Imágenes":
    st.header("Carga de Imágenes Satelitales")
    
    uploaded_files = st.file_uploader(
        "Sube imágenes del lote (PNG, JPG, TIF)",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_images = uploaded_files
        st.success(f"✅ {len(uploaded_files)} imágenes cargadas")
        
        # Mostrar preview
        st.subheader("Vista Previa de Imágenes")
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files[:3]):
            with cols[idx]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_column_width=True)

elif page == "🔍 Detección":
    st.header("Detección de Palmeras")
    
    if not st.session_state.uploaded_images:
        st.warning("Primero carga imágenes en 'Cargar Imágenes'")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuración")
            num_palms = st.slider("Número de palmas a simular", 10, 100, 45)
            detection_confidence = st.slider("Confianza de detección", 0.1, 1.0, 0.8)
            
            if st.button("Ejecutar Detección", type="primary"):
                with st.spinner("Detectando palmas en el lote..."):
                    # Simulación de detección
                    np.random.seed(42)
                    st.session_state.palm_data = [
                        {
                            'id': f'P{i:03d}',
                            'lat': np.random.uniform(4.5, 4.7),
                            'lon': np.random.uniform(-74.2, -74.0),
                            'ndvi': np.random.uniform(0.3, 0.85),
                            'health_status': 'Óptimo' if np.random.random() > 0.3 else 'Estrés'
                        }
                        for i in range(1, num_palms + 1)
                    ]
                    st.success(f"✅ {num_palms} palmas detectadas")
        
        with col2:
            if st.session_state.palm_data:
                st.subheader("Mapa de Detección")
                df = pd.DataFrame(st.session_state.palm_data)
                
                # Crear mapa de calor
                fig = px.density_mapbox(
                    df, 
                    lat='lat', 
                    lon='lon', 
                    z='ndvi',
                    radius=20,
                    center=dict(lat=4.6, lon=-74.1),
                    zoom=10,
                    mapbox_style="open-street-map",
                    title="Densidad de NDVI - Palmeras Detectadas"
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Análisis":
    st.header("Análisis y Reportes")
    
    if not st.session_state.palm_data:
        st.warning("Ejecuta la detección primero para ver análisis")
    else:
        df = pd.DataFrame(st.session_state.palm_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Salud")
            health_count = df['health_status'].value_counts()
            fig1 = px.pie(
                values=health_count.values,
                names=health_count.index,
                title="Estado de Salud de Palmeras"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Estadísticas NDVI")
            fig2 = px.histogram(
                df, 
                x='ndvi', 
                nbins=20,
                title="Distribución de NDVI"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Métricas de análisis
        st.subheader("Métricas del Lote")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NDVI Promedio", f"{df['ndvi'].mean():.3f}")
        with col2:
            st.metric("Palmas Saludables", f"{len(df[df['health_status'] == 'Óptimo'])}")
        with col3:
            st.metric("Tasa de Estrés", f"{(len(df[df['health_status'] == 'Estrés']) / len(df) * 100):.1f}%")
        with col4:
            st.metric("Densidad", f"{(len(df) / (st.session_state.kml_data['area_ha'] if st.session_state.kml_data else 1)):.1f} palmas/ha")

# Footer
st.markdown("---")
st.markdown("🌴 **Sistema de Monitoreo - Carga KML Habilitada**")
