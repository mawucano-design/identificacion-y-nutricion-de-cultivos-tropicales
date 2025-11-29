import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración más básica posible
st.set_page_config(page_title="Monitoreo Palma", layout="centered")

st.title("🌴 Monitoreo de Palma Aceitera")
st.markdown("---")

# Navegación simple
page = st.sidebar.selectbox("Navegación", [
    "Dashboard", 
    "Datos", 
    "Análisis",
    "Polígono"  # Nueva página
])

if page == "Dashboard":
    st.header("Dashboard Principal")
    
    # Métricas básicas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Palmas", "1,247", "+12")
    with col2:
        st.metric("Salud", "92%", "-1%")
    with col3:
        st.metric("NDVI", "0.74", "+0.03")
    
    # Gráfico simple
    data = pd.DataFrame({
        'Mes': ['Ene', 'Feb', 'Mar', 'Abr', 'May'],
        'NDVI': [0.72, 0.73, 0.74, 0.72, 0.74]
    })
    fig = px.line(data, x='Mes', y='NDVI', title='Tendencia NDVI')
    st.plotly_chart(fig)

elif page == "Datos":
    st.header("Gestión de Datos")
    
    uploaded_file = st.file_uploader("Sube archivo CSV o imagen", type=['csv', 'png', 'jpg'])
    
    if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa de datos:")
            st.dataframe(df.head())
        else:
            st.image(uploaded_file, caption="Imagen cargada", width=300)

elif page == "Análisis":
    st.header("Análisis Básico")
    
    # Datos de ejemplo
    data = pd.DataFrame({
        'Palma': [f'P{i:03d}' for i in range(1, 11)],
        'NDVI': [0.82, 0.78, 0.65, 0.72, 0.58, 0.81, 0.79, 0.45, 0.68, 0.75],
        'Estado': ['Óptimo', 'Óptimo', 'Saludable', 'Óptimo', 'Estrés', 'Óptimo', 'Óptimo', 'Estrés Severo', 'Saludable', 'Óptimo']
    })
    
    fig = px.bar(data, x='Palma', y='NDVI', color='Estado', 
                 title="NDVI por Palma")
    st.plotly_chart(fig)

elif page == "Polígono":
    st.header("Cargar Polígono desde KML")
    
    uploaded_kml = st.file_uploader("Carga un archivo KML", type=['kml'])
    
    if uploaded_kml:
        try:
            import xml.etree.ElementTree as ET
            
            def parse_kml(uploaded_file):
                tree = ET.parse(uploaded_file)
                root = tree.getroot()
                
                # Namespace para KML
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                
                # Buscar el polígono
                polygons = []
                for polygon in root.findall('.//kml:Polygon', ns):
                    coords_elem = polygon.find('.//kml:coordinates', ns)
                    if coords_elem is not None:
                        coords_text = coords_elem.text.strip()
                        # Parsear coordenadas: cada línea es "lon,lat,alt"
                        coords_list = []
                        for coord in coords_text.split():
                            parts = coord.split(',')
                            if len(parts) >= 2:
                                lon, lat = float(parts[0]), float(parts[1])
                                coords_list.append([lon, lat])
                        polygons.append(coords_list)
                
                return polygons
            
            polygons = parse_kml(uploaded_kml)
            
            if not polygons:
                st.error("No se encontraron polígonos en el archivo KML.")
            else:
                st.success(f"Se encontraron {len(polygons)} polígono(s).")
                
                # Mostrar el primer polígono en un mapa de Plotly
                coords = polygons[0]
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                
                # Crear un mapa con el polígono
                fig = go.Figure(go.Scattermapbox(
                    mode="lines",
                    lon=lons,
                    lat=lats,
                    marker={'size': 10},
                    line=dict(width=2, color='blue')
                ))
                
                # Ajustar el zoom y centro del mapa
                center_lon = sum(lons) / len(lons)
                center_lat = sum(lats) / len(lats)
                
                fig.update_layout(
                    mapbox={
                        'style': "open-street-map",
                        'center': {'lon': center_lon, 'lat': center_lat},
                        'zoom': 10
                    },
                    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
                )
                
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"Error al cargar el KML: {e}")
            st.info("Asegúrate de que el archivo KML contenga un polígono válido.")

st.markdown("---")
st.markdown("✅ **Sistema funcionando correctamente**")
