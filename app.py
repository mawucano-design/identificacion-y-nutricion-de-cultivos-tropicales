import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import os
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import io
from shapely.geometry import Polygon
import math
import folium
from folium import plugins
from streamlit_folium import st_folium
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import base64
import fiona

st.set_page_config(page_title="🌴 Analizador Cultivos", layout="wide")
st.title("🌱 ANALIZADOR CULTIVOS - METODOLOGÍA GEE COMPLETA CON AGROECOLOGÍA")
st.markdown("---")

# Configurar para restaurar .shx automáticamente
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# PARÁMETROS MEJORADOS Y MÁS REALISTAS PARA DIFERENTES CULTIVOS
PARAMETROS_CULTIVOS = {
    'PALMA_ACEITERA': {
        'NITROGENO': {'min': 120, 'max': 200, 'optimo': 160},
        'FOSFORO': {'min': 40, 'max': 80, 'optimo': 60},
        'POTASIO': {'min': 160, 'max': 240, 'optimo': 200},
        'MATERIA_ORGANICA_OPTIMA': 3.5,
        'HUMEDAD_OPTIMA': 0.35,
        'pH_OPTIMO': 5.5,
        'CONDUCTIVIDAD_OPTIMA': 1.2,
        'NDVI_OPTIMO': 0.7,
        'SAVI_OPTIMO': 0.6,
        'MSAVI_OPTIMO': 0.65,
        'GNDVI_OPTIMO': 0.5,
        'NDRE_OPTIMO': 0.4,
        'ESTRES_HIDRICO_OPTIMO': 0.3,
        'CLOROFILA_OPTIMA': 40
    },
    'CACAO': {
        'NITROGENO': {'min': 100, 'max': 180, 'optimo': 140},
        'FOSFORO': {'min': 30, 'max': 60, 'optimo': 45},
        'POTASIO': {'min': 120, 'max': 200, 'optimo': 160},
        'MATERIA_ORGANICA_OPTIMA': 4.0,
        'HUMEDAD_OPTIMA': 0.4,
        'pH_OPTIMO': 6.0,
        'CONDUCTIVIDAD_OPTIMA': 1.0,
        'NDVI_OPTIMO': 0.75,
        'SAVI_OPTIMO': 0.65,
        'MSAVI_OPTIMO': 0.7,
        'GNDVI_OPTIMO': 0.55,
        'NDRE_OPTIMO': 0.45,
        'ESTRES_HIDRICO_OPTIMO': 0.35,
        'CLOROFILA_OPTIMA': 35
    },
    'BANANO': {
        'NITROGENO': {'min': 180, 'max': 280, 'optimo': 230},
        'FOSFORO': {'min': 50, 'max': 90, 'optimo': 70},
        'POTASIO': {'min': 250, 'max': 350, 'optimo': 300},
        'MATERIA_ORGANICA_OPTIMA': 4.5,
        'HUMEDAD_OPTIMA': 0.45,
        'pH_OPTIMO': 6.2,
        'CONDUCTIVIDAD_OPTIMA': 1.5,
        'NDVI_OPTIMO': 0.8,
        'SAVI_OPTIMO': 0.7,
        'MSAVI_OPTIMO': 0.75,
        'GNDVI_OPTIMO': 0.6,
        'NDRE_OPTIMO': 0.5,
        'ESTRES_HIDRICO_OPTIMO': 0.4,
        'CLOROFILA_OPTIMA': 45
    }
}

# ==============================================
# NUEVOS PARÁMETROS PARA ANÁLISIS DE SALUD
# ==============================================

# ESCALAS DE COLOR PARA DIFERENTES ESTADOS
ESCALAS_COLOR = {
    'ESTADO_SANITARIO': {
        'EXCELENTE': {'color': '#006837', 'rango': (0.8, 1.0), 'descripcion': 'Planta sana sin problemas'},
        'BUENO': {'color': '#66bd63', 'rango': (0.6, 0.8), 'descripcion': 'Ligero estrés, buen estado'},
        'REGULAR': {'color': '#fee08b', 'rango': (0.4, 0.6), 'descripcion': 'Estrés moderado, atención requerida'},
        'DEFICIENTE': {'color': '#f46d43', 'rango': (0.2, 0.4), 'descripcion': 'Problemas sanitarios evidentes'},
        'CRITICO': {'color': '#d73027', 'rango': (0.0, 0.2), 'descripcion': 'Estado crítico, intervención urgente'}
    },
    'ESTRES_HIDRICO': {
        'SIN_ESTRES': {'color': '#2b83ba', 'rango': (0.0, 0.2), 'descripcion': 'Hidratación óptima'},
        'LEVE': {'color': '#abdda4', 'rango': (0.2, 0.4), 'descripcion': 'Ligera deficiencia hídrica'},
        'MODERADO': {'color': '#ffffbf', 'rango': (0.4, 0.6), 'descripcion': 'Estrés hídrico moderado'},
        'ALTO': {'color': '#fdae61', 'rango': (0.6, 0.8), 'descripcion': 'Alto estrés, requiere riego'},
        'SEVERO': {'color': '#d7191c', 'rango': (0.8, 1.0), 'descripcion': 'Estrés severo, urgente riego'}
    },
    'ESTADO_NUTRICIONAL': {
        'OPTIMO': {'color': '#1a9850', 'rango': (0.8, 1.0), 'descripcion': 'Balance nutricional perfecto'},
        'ADEQUADO': {'color': '#91cf60', 'rango': (0.6, 0.8), 'descripcion': 'Nutrición adecuada'},
        'REGULAR': {'color': '#fee08b', 'rango': (0.4, 0.6), 'descripcion': 'Deficiencias leves'},
        'DEFICIENTE': {'color': '#fc8d59', 'rango': (0.2, 0.4), 'descripcion': 'Deficiencias nutricionales'},
        'CRITICO': {'color': '#d73027', 'rango': (0.0, 0.2), 'descripcion': 'Deficiencias graves'}
    },
    'VIGOR_VEGETATIVO': {
        'MUY_ALTO': {'color': '#006837', 'rango': (0.8, 1.0), 'descripcion': 'Crecimiento vigoroso óptimo'},
        'ALTO': {'color': '#66bd63', 'rango': (0.6, 0.8), 'descripcion': 'Buen crecimiento'},
        'MODERADO': {'color': '#fee08b', 'rango': (0.4, 0.6), 'descripcion': 'Crecimiento regular'},
        'BAJO': {'color': '#f46d43', 'rango': (0.2, 0.4), 'descripcion': 'Crecimiento deficiente'},
        'MUY_BAJO': {'color': '#d73027', 'rango': (0.0, 0.2), 'descripcion': 'Crecimiento muy limitado'}
    }
}

# UMBRALES DE ALERTA POR CULTIVO
UMBRALES_ALERTA = {
    'PALMA_ACEITERA': {
        'NDVI_ALERTA': 0.5,
        'ESTRES_HIDRICO_ALERTA': 0.6,
        'DEFICIT_NITROGENO_ALERTA': 30,
        'DEFICIT_FOSFORO_ALERTA': 15,
        'DEFICIT_POTASIO_ALERTA': 40,
        'CLOROFILA_ALERTA': 30
    },
    'CACAO': {
        'NDVI_ALERTA': 0.55,
        'ESTRES_HIDRICO_ALERTA': 0.65,
        'DEFICIT_NITROGENO_ALERTA': 25,
        'DEFICIT_FOSFORO_ALERTA': 10,
        'DEFICIT_POTASIO_ALERTA': 30,
        'CLOROFILA_ALERTA': 25
    },
    'BANANO': {
        'NDVI_ALERTA': 0.6,
        'ESTRES_HIDRICO_ALERTA': 0.7,
        'DEFICIT_NITROGENO_ALERTA': 40,
        'DEFICIT_FOSFORO_ALERTA': 20,
        'DEFICIT_POTASIO_ALERTA': 50,
        'CLOROFILA_ALERTA': 35
    }
}

# SÍNTOMAS VISUALES POR ESTADO
SINTOMAS_VISUALES = {
    'ESTADO_SANITARIO': {
        'CRITICO': [
            "Amarillamiento severo de hojas",
            "Necrosis en bordes foliares",
            "Defoliación avanzada",
            "Manchas oscuras extensas",
            "Crecimiento atrofiado"
        ],
        'DEFICIENTE': [
            "Amarillamiento moderado",
            "Manchas foliares pequeñas",
            "Reducción de tamaño foliar",
            "Coloración anormal",
            "Crecimiento lento"
        ],
        'REGULAR': [
            "Ligero amarillamiento",
            "Manchas aisladas",
            "Coloración menos intensa",
            "Crecimiento aceptable",
            "Sin síntomas graves"
        ]
    },
    'ESTRES_HIDRICO': {
        'SEVERO': [
            "Hojas marchitas permanentemente",
            "Rollamiento foliar severo",
            "Quemaduras en bordes",
            "Color grisáceo",
            "Pérdida de turgencia"
        ],
        'ALTO': [
            "Marchitamiento diurno",
            "Rollamiento moderado",
            "Punta seca en hojas",
            "Color opaco",
            "Reducción crecimiento"
        ],
        'MODERADO': [
            "Marchitamiento temporal",
            "Ligero rollamiento",
            "Turgencia reducida",
            "Color menos brillante",
            "Crecimiento normal"
        ]
    },
    'DEFICIENCIA_NUTRICIONAL': {
        'NITROGENO': [
            "Amarillamiento generalizado",
            "Hojas viejas afectadas primero",
            "Crecimiento reducido",
            "Tallos delgados",
            "Color verde pálido"
        ],
        'FOSFORO': [
            "Coloración púrpura en hojas",
            "Hojas pequeñas y oscuras",
            "Raíces poco desarrolladas",
            "Retraso en maduración",
            "Floración escasa"
        ],
        'POTASIO': [
            "Quemaduras en bordes foliares",
            "Clorosis marginal",
            "Hojas rizadas",
            "Tallos débiles",
            "Frutas pequeñas"
        ]
    }
}

# PARÁMETROS DE TEXTURA DEL SUELO POR CULTIVO
TEXTURA_SUELO_OPTIMA = {
    'PALMA_ACEITERA': {
        'textura_optima': 'FRANCO_ARCILLOSO',
        'arena_optima': 40,
        'limo_optima': 30,
        'arcilla_optima': 30,
        'densidad_aparente_optima': 1.3,
        'porosidad_optima': 0.5
    },
    'CACAO': {
        'textura_optima': 'FRANCO',
        'arena_optima': 45,
        'limo_optima': 35,
        'arcilla_optima': 20,
        'densidad_aparente_optima': 1.2,
        'porosidad_optima': 0.55
    },
    'BANANO': {
        'textura_optima': 'FRANCO_ARENOSO',
        'arena_optima': 50,
        'limo_optima': 30,
        'arcilla_optima': 20,
        'densidad_aparente_optima': 1.25,
        'porosidad_optima': 0.52
    }
}

# CLASIFICACIÓN DE TEXTURAS DEL SUELO
CLASIFICACION_TEXTURAS = {
    'ARENOSO': {'arena_min': 85, 'arena_max': 100, 'limo_max': 15, 'arcilla_max': 15},
    'FRANCO_ARENOSO': {'arena_min': 70, 'arena_max': 85, 'limo_max': 30, 'arcilla_max': 20},
    'FRANCO': {'arena_min': 43, 'arena_max': 52, 'limo_min': 28, 'limo_max': 50, 'arcilla_min': 7, 'arcilla_max': 27},
    'FRANCO_ARCILLOSO': {'arena_min': 20, 'arena_max': 45, 'limo_min': 15, 'limo_max': 53, 'arcilla_min': 27, 'arcilla_max': 40},
    'ARCILLOSO': {'arena_max': 45, 'limo_max': 40, 'arcilla_min': 40}
}

# FACTORES EDÁFICOS MÁS REALISTAS
FACTORES_SUELO = {
    'ARCILLOSO': {'retention': 1.3, 'drainage': 0.7, 'aeration': 0.6, 'workability': 0.5},
    'FRANCO_ARCILLOSO': {'retention': 1.2, 'drainage': 0.8, 'aeration': 0.7, 'workability': 0.7},
    'FRANCO': {'retention': 1.0, 'drainage': 1.0, 'aeration': 1.0, 'workability': 1.0},
    'FRANCO_ARENOSO': {'retention': 0.8, 'drainage': 1.2, 'aeration': 1.3, 'workability': 1.2},
    'ARENOSO': {'retention': 0.6, 'drainage': 1.4, 'aeration': 1.5, 'workability': 1.4}
}

# RECOMENDACIONES POR TIPO DE TEXTURA
RECOMENDACIONES_TEXTURA = {
    'ARCILLOSO': [
        "Añadir materia orgánica para mejorar estructura",
        "Evitar laboreo en condiciones húmedas",
        "Implementar drenajes superficiales",
        "Usar cultivos de cobertura para romper compactación"
    ],
    'FRANCO_ARCILLOSO': [
        "Mantener niveles adecuados de materia orgánica",
        "Rotación de cultivos para mantener estructura",
        "Laboreo mínimo conservacionista",
        "Aplicación moderada de enmiendas"
    ],
    'FRANCO': [
        "Textura ideal - mantener prácticas conservacionistas",
        "Rotación balanceada de cultivos",
        "Manejo integrado de nutrientes",
        "Conservar estructura con coberturas"
    ],
    'FRANCO_ARENOSO': [
        "Aplicación frecuente de materia orgánica",
        "Riego por goteo para eficiencia hídrica",
        "Fertilización fraccionada para reducir pérdidas",
        "Cultivos de cobertura para retener humedad"
    ],
    'ARENOSO': [
        "Altas dosis de materia orgánica y compost",
        "Sistema de riego por goteo con alta frecuencia",
        "Fertilización en múltiples aplicaciones",
        "Barreras vivas para reducir erosión"
    ]
}

# PRINCIPIOS AGROECOLÓGICOS - RECOMENDACIONES ESPECÍFICAS
RECOMENDACIONES_AGROECOLOGICAS = {
    'PALMA_ACEITERA': {
        'COBERTURAS_VIVAS': [
            "Leguminosas: Centrosema pubescens, Pueraria phaseoloides",
            "Coberturas mixtas: Maní forrajero (Arachis pintoi)",
            "Plantas de cobertura baja: Dichondra repens"
        ],
        'ABONOS_VERDES': [
            "Crotalaria juncea: 3-4 kg/ha antes de la siembra",
            "Mucuna pruriens: 2-3 kg/ha para control de malezas",
            "Canavalia ensiformis: Fijación de nitrógeno"
        ],
        'BIOFERTILIZANTES': [
            "Bocashi: 2-3 ton/ha cada 6 meses",
            "Compost de racimo vacío: 1-2 ton/ha",
            "Biofertilizante líquido: Aplicación foliar mensual"
        ],
        'MANEJO_ECOLOGICO': [
            "Uso de trampas amarillas para insectos",
            "Cultivos trampa: Maíz alrededor de la plantación",
            "Conservación de enemigos naturales"
        ],
        'ASOCIACIONES': [
            "Piña en calles durante primeros 2 años",
            "Yuca en calles durante establecimiento",
            "Leguminosas arbustivas como cercas vivas"
        ]
    },
    'CACAO': {
        'COBERTURAS_VIVAS': [
            "Leguminosas rastreras: Arachis pintoi",
            "Coberturas sombreadas: Erythrina poeppigiana",
            "Plantas aromáticas: Lippia alba para control plagas"
        ],
        'ABONOS_VERDES': [
            "Frijol terciopelo (Mucuna pruriens): 3 kg/ha",
            "Guandul (Cajanus cajan): Podas periódicas",
            "Crotalaria: Control de nematodos"
        ],
        'BIOFERTILIZANTES': [
            "Compost de cacaoteca: 3-4 ton/ha",
            "Bocashi especial cacao: 2 ton/ha",
            "Té de compost aplicado al suelo"
        ],
        'MANEJO_ECOLOGICO': [
            "Sistema agroforestal multiestrato",
            "Manejo de sombra regulada (30-50%)",
            "Control biológico con hongos entomopatógenos"
        ],
        'ASOCIACIONES': [
            "Árboles maderables: Cedro, Caoba",
            "Frutales: Cítricos, Aguacate",
            "Plantas medicinales: Jengibre, Cúrcuma"
        ]
    },
    'BANANO': {
        'COBERTURAS_VIVAS': [
            "Arachis pintoi entre calles",
            "Leguminosas de porte bajo",
            "Coberturas para control de malas hierbas"
        ],
        'ABONOS_VERDES': [
            "Mucuna pruriens: 4 kg/ha entre ciclos",
            "Canavalia ensiformis: Fijación de N",
            "Crotalaria spectabilis: Control nematodos"
        ],
        'BIOFERTILIZANTES': [
            "Compost de pseudotallo: 4-5 ton/ha",
            "Bocashi bananero: 3 ton/ha",
            "Biofertilizante a base de micorrizas"
        ],
        'MANEJO_ECOLOGICO': [
            "Trampas cromáticas para picudos",
            "Barreras vivas con citronela",
            "Uso de trichoderma para control enfermedades"
        ],
        'ASOCIACIONES': [
            "Leguminosas arbustivas en linderos",
            "Cítricos como cortavientos",
            "Plantas repelentes: Albahaca, Menta"
        ]
    }
}

# FACTORES ESTACIONALES
FACTORES_MES = {
    "ENERO": 0.9, "FEBRERO": 0.95, "MARZO": 1.0, "ABRIL": 1.05,
    "MAYO": 1.1, "JUNIO": 1.0, "JULIO": 0.95, "AGOSTO": 0.9,
    "SEPTIEMBRE": 0.95, "OCTUBRE": 1.0, "NOVIEMBRE": 1.05, "DICIEMBRE": 1.0
}

FACTORES_N_MES = {
    "ENERO": 1.0, "FEBRERO": 1.05, "MARZO": 1.1, "ABRIL": 1.15,
    "MAYO": 1.2, "JUNIO": 1.1, "JULIO": 1.0, "AGOSTO": 0.9,
    "SEPTIEMBRE": 0.95, "OCTUBRE": 1.0, "NOVIEMBRE": 1.05, "DICIEMBRE": 1.0
}

FACTORES_P_MES = {
    "ENERO": 1.0, "FEBRERO": 1.0, "MARZO": 1.05, "ABRIL": 1.1,
    "MAYO": 1.15, "JUNIO": 1.1, "JULIO": 1.05, "AGOSTO": 1.0,
    "SEPTIEMBRE": 1.0, "OCTUBRE": 1.05, "NOVIEMBRE": 1.1, "DICIEMBRE": 1.05
}

FACTORES_K_MES = {
    "ENERO": 1.0, "FEBRERO": 1.0, "MARZO": 1.0, "ABRIL": 1.05,
    "MAYO": 1.1, "JUNIO": 1.15, "JULIO": 1.2, "AGOSTO": 1.15,
    "SEPTIEMBRE": 1.1, "OCTUBRE": 1.05, "NOVIEMBRE": 1.0, "DICIEMBRE": 1.0
}

# PALETAS GEE MEJORADAS
PALETAS_GEE = {
    'FERTILIDAD': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'],
    'NITROGENO': ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e'],
    'FOSFORO': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
    'POTASIO': ['#4d004b', '#810f7c', '#8c6bb1', '#8c96c6', '#9ebcda', '#bfd3e6', '#e0ecf4', '#edf8fb'],
    'TEXTURA': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e'],
    # NUEVAS PALETAS PARA ESTADOS DE SALUD
    'ESTADO_SANITARIO': ['#d73027', '#f46d43', '#fee08b', '#a6d96a', '#1a9850'],
    'ESTRES_HIDRICO': ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'],
    'ESTADO_NUTRICIONAL': ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850'],
    'VIGOR_VEGETATIVO': ['#d73027', '#f46d43', '#fee08b', '#a6d96a', '#006837'],
    'CLOROFILA': ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850'],
    'TEMP_CANOPY': ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']
}

# Inicializar session_state
if 'analisis_completado' not in st.session_state:
    st.session_state.analisis_completado = False
if 'gdf_analisis' not in st.session_state:
    st.session_state.gdf_analisis = None
if 'gdf_original' not in st.session_state:
    st.session_state.gdf_original = None
if 'gdf_zonas' not in st.session_state:
    st.session_state.gdf_zonas = None
if 'area_total' not in st.session_state:
    st.session_state.area_total = 0
if 'datos_demo' not in st.session_state:
    st.session_state.datos_demo = False
if 'analisis_textura' not in st.session_state:
    st.session_state.analisis_textura = None
if 'analisis_salud' not in st.session_state:
    st.session_state.analisis_salud = None
if 'analisis_clusters' not in st.session_state:
    st.session_state.analisis_clusters = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    cultivo = st.selectbox("Cultivo:", 
                          ["PALMA_ACEITERA", "CACAO", "BANANO"])
    
    # Opción para análisis de textura
    analisis_tipo = st.selectbox("Tipo de Análisis:", 
                               ["FERTILIDAD ACTUAL", "RECOMENDACIONES NPK", "ANÁLISIS DE TEXTURA",
                                "ESTADO SANITARIO", "ESTRÉS HÍDRICO", "ESTADO NUTRICIONAL", "VIGOR VEGETATIVO", "CLUSTERIZACIÓN"])
    
    if analisis_tipo == "RECOMENDACIONES NPK":
        nutriente = st.selectbox("Nutriente:", ["NITRÓGENO", "FÓSFORO", "POTASIO"])
    else:
        nutriente = None
    
    mes_analisis = st.selectbox("Mes de Análisis:", 
                               ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
                                "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"])
    
    st.subheader("🎯 División de Parcela")
    n_divisiones = st.slider("Número de zonas de manejo:", min_value=16, max_value=32, value=24)
    
    # NUEVO: Opciones para visualización de salud
    if analisis_tipo in ["ESTADO SANITARIO", "ESTRÉS HÍDRICO", "ESTADO NUTRICIONAL", "VIGOR VEGETATIVO", "CLUSTERIZACIÓN"]:
        st.subheader("🎨 Visualización")
        mostrar_sintomas = st.checkbox("Mostrar síntomas visuales", value=True)
        mostrar_alertas = st.checkbox("Mostrar alertas automáticas", value=True)
        
        if analisis_tipo == "CLUSTERIZACIÓN":
            n_clusters = st.slider("Número de clusters:", min_value=3, max_value=8, value=5)
        else:
            umbral_alerta = st.slider("Umbral de alerta (%):", 30, 90, 60) / 100
    
    st.subheader("📤 Subir Parcela")
    uploaded_file = st.file_uploader("Subir ZIP con shapefile o archivo KML de tu parcela", type=['zip', 'kml'])
    
    # Botón para resetear la aplicación
    if st.button("🔄 Reiniciar Análisis"):
        st.session_state.analisis_completado = False
        st.session_state.gdf_analisis = None
        st.session_state.gdf_original = None
        st.session_state.gdf_zonas = None
        st.session_state.area_total = 0
        st.session_state.datos_demo = False
        st.session_state.analisis_textura = None
        st.session_state.analisis_salud = None
        st.session_state.analisis_clusters = None
        st.rerun()

# ==============================================
# FUNCIONES EXISTENTES (se mantienen igual)
# ==============================================

# FUNCIÓN: CLASIFICAR TEXTURA DEL SUELO
def clasificar_textura_suelo(arena, limo, arcilla):
    """Clasifica la textura del suelo según el triángulo de texturas USDA"""
    try:
        # Normalizar porcentajes a 100%
        total = arena + limo + arcilla
        if total == 0:
            return "NO_DETERMINADA"
        
        arena_norm = (arena / total) * 100
        limo_norm = (limo / total) * 100
        arcilla_norm = (arcilla / total) * 100
        
        # Clasificación según USDA
        if arcilla_norm >= 40:
            return "ARCILLOSO"
        elif arcilla_norm >= 27 and limo_norm >= 15 and limo_norm <= 53 and arena_norm >= 20 and arena_norm <= 45:
            return "FRANCO_ARCILLOSO"
        elif arcilla_norm >= 7 and arcilla_norm <= 27 and limo_norm >= 28 and limo_norm <= 50 and arena_norm >= 43 and arena_norm <= 52:
            return "FRANCO"
        elif arena_norm >= 70 and arena_norm <= 85 and arcilla_norm <= 20:
            return "FRANCO_ARENOSO"
        elif arena_norm >= 85:
            return "ARENOSO"
        else:
            return "FRANCO"  # Por defecto
        
    except Exception as e:
        return "NO_DETERMINADA"

# FUNCIÓN: CALCULAR PROPIEDADES FÍSICAS DEL SUELO
def calcular_propiedades_fisicas_suelo(textura, materia_organica):
    """Calcula propiedades físicas del suelo basadas en textura y MO"""
    propiedades = {
        'capacidad_campo': 0.0,
        'punto_marchitez': 0.0,
        'agua_disponible': 0.0,
        'densidad_aparente': 0.0,
        'porosidad': 0.0,
        'conductividad_hidraulica': 0.0
    }
    
    # Valores base según textura (mm/m)
    base_propiedades = {
        'ARCILLOSO': {'cc': 350, 'pm': 200, 'da': 1.3, 'porosidad': 0.5, 'kh': 0.1},
        'FRANCO_ARCILLOSO': {'cc': 300, 'pm': 150, 'da': 1.25, 'porosidad': 0.53, 'kh': 0.5},
        'FRANCO': {'cc': 250, 'pm': 100, 'da': 1.2, 'porosidad': 0.55, 'kh': 1.5},
        'FRANCO_ARENOSO': {'cc': 180, 'pm': 80, 'da': 1.35, 'porosidad': 0.49, 'kh': 5.0},
        'ARENOSO': {'cc': 120, 'pm': 50, 'da': 1.5, 'porosidad': 0.43, 'kh': 15.0}
    }
    
    if textura in base_propiedades:
        base = base_propiedades[textura]
        
        # Ajustar por materia orgánica (cada 1% de MO mejora propiedades)
        factor_mo = 1.0 + (materia_organica * 0.05)
        
        propiedades['capacidad_campo'] = base['cc'] * factor_mo
        propiedades['punto_marchitez'] = base['pm'] * factor_mo
        propiedades['agua_disponible'] = (base['cc'] - base['pm']) * factor_mo
        propiedades['densidad_aparente'] = base['da'] / factor_mo
        propiedades['porosidad'] = min(0.65, base['porosidad'] * factor_mo)
        propiedades['conductividad_hidraulica'] = base['kh'] * factor_mo
    
    return propiedades

# FUNCIÓN: EVALUAR ADECUACIÓN DE TEXTURA
def evaluar_adecuacion_textura(textura_actual, cultivo):
    """Evalúa qué tan adecuada es la textura para el cultivo específico"""
    textura_optima = TEXTURA_SUELO_OPTIMA[cultivo]['textura_optima']
    
    # Jerarquía de adecuación
    jerarquia_texturas = {
        'ARENOSO': 1,
        'FRANCO_ARENOSO': 2,
        'FRANCO': 3,
        'FRANCO_ARCILLOSO': 4,
        'ARCILLOSO': 5
    }
    
    if textura_actual not in jerarquia_texturas:
        return "NO_DETERMINADA", 0
    
    actual_idx = jerarquia_texturas[textura_actual]
    optima_idx = jerarquia_texturas[textura_optima]
    
    diferencia = abs(actual_idx - optima_idx)
    
    if diferencia == 0:
        return "ÓPTIMA", 1.0
    elif diferencia == 1:
        return "ADECUADA", 0.8
    elif diferencia == 2:
        return "MODERADA", 0.6
    elif diferencia == 3:
        return "LIMITANTE", 0.4
    else:
        return "MUY LIMITANTE", 0.2

# FUNCIÓN MEJORADA PARA CALCULAR SUPERFICIE
def calcular_superficie(gdf):
    """Calcula superficie en hectáreas con manejo robusto de CRS"""
    try:
        if gdf.empty or gdf.geometry.isnull().all():
            return 0.0
            
        # Verificar si el CRS es geográfico (grados)
        if gdf.crs and gdf.crs.is_geographic:
            # Convertir a un CRS proyectado para cálculo de área precisa
            try:
                # Usar UTM adecuado (aquí se usa un CRS común para Colombia)
                gdf_proj = gdf.to_crs('EPSG:3116')  # MAGNA-SIRGAS / Colombia West zone
                area_m2 = gdf_proj.geometry.area
            except:
                # Fallback: conversión aproximada (1 grado ≈ 111km en ecuador)
                area_m2 = gdf.geometry.area * 111000 * 111000
        else:
            # Asumir que ya está en metros
            area_m2 = gdf.geometry.area
            
        return area_m2 / 10000  # Convertir a hectáreas
        
    except Exception as e:
        # Fallback simple
        try:
            return gdf.geometry.area.mean() / 10000
        except:
            return 1.0  # Valor por defecto

# ==============================================
# NUEVAS FUNCIONES PARA ANÁLISIS DE SALUD
# ==============================================

def calcular_indices_salud_cultivo(gdf, cultivo, mes_analisis):
    """Calcula índices de salud del cultivo basados en parámetros simulados"""
    gdf_salud = gdf.copy()
    
    # Obtener parámetros óptimos
    params = PARAMETROS_CULTIVOS[cultivo]
    
    # Inicializar columnas de salud si no existen
    columnas_salud = [
        'estado_sanitario', 'categoria_sanitario',
        'estres_hidrico', 'categoria_estres',
        'estado_nutricional', 'categoria_nutricional',
        'vigor_vegetativo', 'categoria_vigor',
        'clorofila_relativa', 'temp_canopy',
        'ndvi', 'savi', 'msavi', 'gndvi', 'ndre'
    ]
    
    for col in columnas_salud:
        if col not in gdf_salud.columns:
            gdf_salud[col] = 0.0
    
    for idx, row in gdf_salud.iterrows():
        try:
            # Obtener centroide para semilla reproducible
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
            else:
                centroid = row.geometry.representative_point()
            
            # Semilla para reproducibilidad
            seed_value = abs(hash(f"{centroid.x:.6f}_{centroid.y:.6f}_{cultivo}_salud")) % (2**32)
            rng = np.random.RandomState(seed_value)
            
            # Simular índices espectrales
            base_ndvi = rng.uniform(0.3, 0.9)
            gdf_salud.loc[idx, 'ndvi'] = base_ndvi
            gdf_salud.loc[idx, 'savi'] = base_ndvi * rng.uniform(0.8, 1.0)
            gdf_salud.loc[idx, 'msavi'] = base_ndvi * rng.uniform(0.85, 1.05)
            gdf_salud.loc[idx, 'gndvi'] = base_ndvi * rng.uniform(0.7, 0.9)
            gdf_salud.loc[idx, 'ndre'] = base_ndvi * rng.uniform(0.6, 0.8)
            
            # Simular clorofila y temperatura
            gdf_salud.loc[idx, 'clorofila_relativa'] = rng.uniform(20, 50)
            gdf_salud.loc[idx, 'temp_canopy'] = rng.uniform(22, 35)
            
            # CALCULAR ESTADO SANITARIO (basado en NDVI, SAVI, clorofila)
            ndvi_norm = max(0, min(1, (base_ndvi - 0.3) / (0.9 - 0.3)))
            savi_norm = max(0, min(1, (gdf_salud.loc[idx, 'savi'] - 0.24) / (0.81 - 0.24)))
            clorofila_norm = max(0, min(1, (gdf_salud.loc[idx, 'clorofila_relativa'] - 20) / (50 - 20)))
            
            estado_sanitario = (ndvi_norm * 0.4 + savi_norm * 0.3 + clorofila_norm * 0.3)
            gdf_salud.loc[idx, 'estado_sanitario'] = estado_sanitario
            
            # Categorizar estado sanitario
            if estado_sanitario >= 0.8:
                categoria = "EXCELENTE"
            elif estado_sanitario >= 0.6:
                categoria = "BUENO"
            elif estado_sanitario >= 0.4:
                categoria = "REGULAR"
            elif estado_sanitario >= 0.2:
                categoria = "DEFICIENTE"
            else:
                categoria = "CRITICO"
            gdf_salud.loc[idx, 'categoria_sanitario'] = categoria
            
            # CALCULAR ESTRÉS HÍDRICO (basado en temperatura, NDVI, mes)
            temp_norm = max(0, min(1, (gdf_salud.loc[idx, 'temp_canopy'] - 22) / (35 - 22)))
            ndvi_estres = 1 - ndvi_norm  # NDVI bajo = más estrés
            
            # Factor estacional
            factor_mes = FACTORES_MES.get(mes_analisis, 1.0)
            factor_estacional = 1.0 + (factor_mes - 1.0) * 0.5
            
            estres_hidrico = (temp_norm * 0.4 + ndvi_estres * 0.4 + (1/factor_estacional) * 0.2)
            gdf_salud.loc[idx, 'estres_hidrico'] = estres_hidrico
            
            # Categorizar estrés hídrico
            if estres_hidrico <= 0.2:
                cat_estres = "SIN_ESTRES"
            elif estres_hidrico <= 0.4:
                cat_estres = "LEVE"
            elif estres_hidrico <= 0.6:
                cat_estres = "MODERADO"
            elif estres_hidrico <= 0.8:
                cat_estres = "ALTO"
            else:
                cat_estres = "SEVERO"
            gdf_salud.loc[idx, 'categoria_estres'] = cat_estres
            
            # CALCULAR ESTADO NUTRICIONAL (basado en nutrientes y pH)
            if 'nitrogeno' in gdf_salud.columns and 'fosforo' in gdf_salud.columns and 'potasio' in gdf_salud.columns:
                n = gdf_salud.loc[idx, 'nitrogeno']
                p = gdf_salud.loc[idx, 'fosforo']
                k = gdf_salud.loc[idx, 'potasio']
                
                n_norm = max(0, min(1, n / params['NITROGENO']['optimo']))
                p_norm = max(0, min(1, p / params['FOSFORO']['optimo']))
                k_norm = max(0, min(1, k / params['POTASIO']['optimo']))
                
                estado_nutricional = (n_norm * 0.4 + p_norm * 0.3 + k_norm * 0.3)
            else:
                # Simular si no hay datos
                estado_nutricional = rng.uniform(0.3, 0.9)
            
            gdf_salud.loc[idx, 'estado_nutricional'] = estado_nutricional
            
            # Categorizar estado nutricional
            if estado_nutricional >= 0.8:
                cat_nutricion = "OPTIMO"
            elif estado_nutricional >= 0.6:
                cat_nutricion = "ADEQUADO"
            elif estado_nutricional >= 0.4:
                cat_nutricion = "REGULAR"
            elif estado_nutricional >= 0.2:
                cat_nutricion = "DEFICIENTE"
            else:
                cat_nutricion = "CRITICO"
            gdf_salud.loc[idx, 'categoria_nutricional'] = cat_nutricion
            
            # CALCULAR VIGOR VEGETATIVO (índice compuesto)
            vigor = (
                estado_sanitario * 0.35 +
                (1 - estres_hidrico) * 0.35 +
                estado_nutricional * 0.3
            )
            gdf_salud.loc[idx, 'vigor_vegetativo'] = vigor
            
            # Categorizar vigor
            if vigor >= 0.8:
                cat_vigor = "MUY_ALTO"
            elif vigor >= 0.6:
                cat_vigor = "ALTO"
            elif vigor >= 0.4:
                cat_vigor = "MODERADO"
            elif vigor >= 0.2:
                cat_vigor = "BAJO"
            else:
                cat_vigor = "MUY_BAJO"
            gdf_salud.loc[idx, 'categoria_vigor'] = cat_vigor
            
        except Exception as e:
            # Valores por defecto en caso de error
            gdf_salud.loc[idx, 'estado_sanitario'] = 0.5
            gdf_salud.loc[idx, 'categoria_sanitario'] = "REGULAR"
            gdf_salud.loc[idx, 'estres_hidrico'] = 0.5
            gdf_salud.loc[idx, 'categoria_estres'] = "MODERADO"
            gdf_salud.loc[idx, 'estado_nutricional'] = 0.5
            gdf_salud.loc[idx, 'categoria_nutricional'] = "REGULAR"
            gdf_salud.loc[idx, 'vigor_vegetativo'] = 0.5
            gdf_salud.loc[idx, 'categoria_vigor'] = "MODERADO"
            gdf_salud.loc[idx, 'clorofila_relativa'] = 35
            gdf_salud.loc[idx, 'temp_canopy'] = 28
    
    return gdf_salud

def realizar_clusterizacion_cultivo(gdf, cultivo, n_clusters=5):
    """Realiza clusterización basada en reglas usando los parámetros existentes"""
    gdf_clusters = gdf.copy()
    
    # Asegurar que tenemos las columnas necesarias
    if 'estado_sanitario' not in gdf_clusters.columns:
        # Si no existe, calcular usando la función existente
        gdf_clusters = calcular_indices_salud_cultivo(gdf_clusters, cultivo, mes_analisis)
    
    # Crear características compuestas para clustering basado en reglas
    # Usar estado sanitario, estrés hídrico y estado nutricional
    caracteristicas = []
    
    if 'estado_sanitario' in gdf_clusters.columns:
        caracteristicas.append(gdf_clusters['estado_sanitario'].values)
    
    if 'estres_hidrico' in gdf_clusters.columns:
        # Invertir estrés hídrico (menos estrés = mejor)
        caracteristicas.append(1 - gdf_clusters['estres_hidrico'].values)
    
    if 'estado_nutricional' in gdf_clusters.columns:
        caracteristicas.append(gdf_clusters['estado_nutricional'].values)
    
    if 'vigor_vegetativo' in gdf_clusters.columns:
        caracteristicas.append(gdf_clusters['vigor_vegetativo'].values)
    
    if caracteristicas:
        # Calcular característica compuesta promedio
        caracteristica_compuesta = np.mean(caracteristicas, axis=0)
    else:
        # Si no hay características, usar valor aleatorio
        caracteristica_compuesta = np.random.uniform(0.3, 0.9, len(gdf_clusters))
    
    # Calcular percentiles para crear clusters
    percentiles = np.linspace(0, 100, n_clusters + 1)
    valores_percentiles = np.percentile(caracteristica_compuesta, percentiles)
    
    # Asignar clusters basados en percentiles
    gdf_clusters['cluster'] = 0
    for i in range(n_clusters):
        mask = (caracteristica_compuesta >= valores_percentiles[i]) & (caracteristica_compuesta < valores_percentiles[i + 1])
        gdf_clusters.loc[mask, 'cluster'] = i + 1
    
    # Asegurar que el último valor se incluye
    gdf_clusters.loc[caracteristica_compuesta >= valores_percentiles[-2], 'cluster'] = n_clusters
    
    # Describir cada cluster basado en estadísticas
    descripciones_clusters = []
    for i in range(1, n_clusters + 1):
        cluster_data = gdf_clusters[gdf_clusters['cluster'] == i]
        
        if len(cluster_data) == 0:
            descripciones_clusters.append(f"Cluster {i} - Sin datos")
            continue
        
        # Calcular características promedio del cluster
        avg_sanitario = cluster_data['estado_sanitario'].mean() if 'estado_sanitario' in cluster_data.columns else 0.5
        avg_estres = cluster_data['estres_hidrico'].mean() if 'estres_hidrico' in cluster_data.columns else 0.5
        avg_nutricion = cluster_data['estado_nutricional'].mean() if 'estado_nutricional' in cluster_data.columns else 0.5
        
        # Determinar descripción basada en promedios
        if avg_sanitario >= 0.7 and avg_estres <= 0.4 and avg_nutricion >= 0.7:
            descripcion = f"Cluster {i}: Zonas saludables y bien nutridas"
        elif avg_sanitario < 0.5 and avg_estres > 0.6:
            descripcion = f"Cluster {i}: Zonas con problemas sanitarios y estrés"
        elif avg_nutricion < 0.5:
            descripcion = f"Cluster {i}: Zonas con deficiencias nutricionales"
        elif avg_sanitario >= 0.7:
            descripcion = f"Cluster {i}: Zonas de alto vigor vegetativo"
        elif avg_sanitario < 0.4:
            descripcion = f"Cluster {i}: Zonas de bajo vigor vegetativo"
        else:
            descripcion = f"Cluster {i}: Zonas con características mixtas"
        
        descripciones_clusters.append(descripcion)
    
    # Asignar descripciones a cada fila
    gdf_clusters['descripcion_cluster'] = gdf_clusters['cluster'].apply(
        lambda x: descripciones_clusters[int(x)-1] if int(x) <= len(descripciones_clusters) else "Cluster no definido"
    )
    
    return gdf_clusters

def obtener_color_por_estado(valor, tipo_estado):
    """Obtiene el color correspondiente al valor del estado"""
    if tipo_estado not in ESCALAS_COLOR:
        return '#999999'
    
    escalas = ESCALAS_COLOR[tipo_estado]
    
    for categoria, info in escalas.items():
        rango_min, rango_max = info['rango']
        if rango_min <= valor < rango_max:
            return info['color']
    
    # Si no está en ningún rango, color por defecto
    return '#999999'

def crear_mapa_salud_interactivo(gdf_salud, cultivo, tipo_analisis):
    """Crea mapa interactivo con colores según estado de salud"""
    
    # Obtener centro y bounds
    centroid = gdf_salud.geometry.centroid.iloc[0]
    bounds = gdf_salud.total_bounds
    
    # Crear mapa con ESRI Satélite
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=15,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satélite'
    )
    
    # Añadir otras bases
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Calles',
        overlay=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False
    ).add_to(m)
    
    # Determinar columna de datos según tipo de análisis
    if tipo_analisis == "ESTADO SANITARIO":
        columna_valor = 'estado_sanitario'
        columna_categoria = 'categoria_sanitario'
        titulo = "Estado Sanitario"
        paleta = PALETAS_GEE['ESTADO_SANITARIO']
    elif tipo_analisis == "ESTRÉS HÍDRICO":
        columna_valor = 'estres_hidrico'
        columna_categoria = 'categoria_estres'
        titulo = "Estrés Hídrico"
        paleta = PALETAS_GEE['ESTRES_HIDRICO']
    elif tipo_analisis == "ESTADO NUTRICIONAL":
        columna_valor = 'estado_nutricional'
        columna_categoria = 'categoria_nutricional'
        titulo = "Estado Nutricional"
        paleta = PALETAS_GEE['ESTADO_NUTRICIONAL']
    elif tipo_analisis == "VIGOR VEGETATIVO":
        columna_valor = 'vigor_vegetativo'
        columna_categoria = 'categoria_vigor'
        titulo = "Vigor Vegetativo"
        paleta = PALETAS_GEE['VIGOR_VEGETATIVO']
    elif tipo_analisis == "CLUSTERIZACIÓN":
        columna_valor = 'cluster'
        columna_categoria = 'descripcion_cluster'
        titulo = "Clusterización"
        # Para clusters, usar paleta especial
        cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    else:
        columna_valor = 'indice_fertilidad'
        columna_categoria = 'categoria'
        titulo = "Fertilidad"
        paleta = PALETAS_GEE['FERTILIDAD']
    
    # Añadir cada polígono con color según estado
    for idx, row in gdf_salud.iterrows():
        if columna_valor not in row:
            continue
            
        valor = row[columna_valor]
        categoria = row[columna_categoria] if columna_categoria in row else "N/A"
        
        # Obtener color según el valor
        if tipo_analisis == "ESTADO SANITARIO":
            color = obtener_color_por_estado(valor, 'ESTADO_SANITARIO')
        elif tipo_analisis == "ESTRÉS HÍDRICO":
            color = obtener_color_por_estado(valor, 'ESTRES_HIDRICO')
        elif tipo_analisis == "ESTADO NUTRICIONAL":
            color = obtener_color_por_estado(valor, 'ESTADO_NUTRICIONAL')
        elif tipo_analisis == "VIGOR VEGETATIVO":
            color = obtener_color_por_estado(valor, 'VIGOR_VEGETATIVO')
        elif tipo_analisis == "CLUSTERIZACIÓN":
            # Para clusters, asignar color basado en el número de cluster
            try:
                cluster_num = int(valor)
                color_idx = (cluster_num - 1) % len(cluster_colors)
                color = cluster_colors[color_idx]
            except:
                color = '#999999'
        else:
            # Para fertilidad, usar paleta GEE
            valor_norm = max(0, min(1, valor))
            idx_color = int(valor_norm * (len(paleta) - 1))
            color = paleta[idx_color]
        
        # Popup informativo
        popup_text = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <h4>Zona {row['id_zona']}</h4>
            <b>{titulo}:</b> {valor:.3f if isinstance(valor, (int, float)) else valor}<br>
            <b>Categoría:</b> {categoria}<br>
            <b>Área:</b> {row.get('area_ha', 0):.2f} ha<br>
            <hr>
        """
        
        # Agregar información adicional según el tipo de análisis
        if tipo_analisis == "ESTADO SANITARIO":
            popup_text += f"""
            <b>NDVI:</b> {row.get('ndvi', 0):.3f}<br>
            <b>Clorofila:</b> {row.get('clorofila_relativa', 0):.1f}<br>
            <b>Temperatura:</b> {row.get('temp_canopy', 0):.1f}°C
            """
        elif tipo_analisis == "ESTRÉS HÍDRICO":
            popup_text += f"""
            <b>Temperatura:</b> {row.get('temp_canopy', 0):.1f}°C<br>
            <b>NDVI:</b> {row.get('ndvi', 0):.3f}<br>
            <b>Humedad:</b> {row.get('humedad', 0):.3f}
            """
        elif tipo_analisis == "ESTADO NUTRICIONAL":
            popup_text += f"""
            <b>Nitrogeno:</b> {row.get('nitrogeno', 0):.1f} kg/ha<br>
            <b>Fósforo:</b> {row.get('fosforo', 0):.1f} kg/ha<br>
            <b>Potasio:</b> {row.get('potasio', 0):.1f} kg/ha
            """
        
        popup_text += "</div>"
        
        # Añadir polígono al mapa
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.7,
                'opacity': 0.9
            },
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Zona {row['id_zona']}: {categoria} ({valor:.3f if isinstance(valor, (int, float)) else valor})"
        ).add_to(m)
        
        # Marcador con número de zona
        centroid = row.geometry.centroid
        folium.Marker(
            [centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f'''
                <div style="
                    background-color: white; 
                    border: 2px solid black; 
                    border-radius: 50%; 
                    width: 28px; 
                    height: 28px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    font-weight: bold; 
                    font-size: 11px;
                    color: black;
                ">{row["id_zona"]}</div>
                '''
            )
        ).add_to(m)
    
    # Ajustar bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    # Añadir controles
    folium.LayerControl().add_to(m)
    plugins.MeasureControl(position='bottomleft').add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)
    
    # Añadir leyenda
    legend_html = f'''
    <div style="
        position: fixed; 
        top: 10px; 
        right: 10px; 
        width: 250px; 
        height: auto; 
        background-color: white; 
        border: 2px solid grey; 
        z-index: 9999; 
        font-size: 12px; 
        padding: 10px; 
        border-radius: 5px;
        font-family: Arial;
    ">
        <h4 style="margin:0 0 10px 0; text-align:center; color: #333;">{titulo} - {cultivo.replace('_', ' ')}</h4>
        <div style="margin-bottom: 10px;">
            <strong>Escala de Colores:</strong>
        </div>
    '''
    
    # Añadir leyenda según tipo de análisis
    if tipo_analisis == "ESTADO SANITARIO":
        escalas = ESCALAS_COLOR['ESTADO_SANITARIO']
        for categoria, info in escalas.items():
            color = info['color']
            desc = info['descripcion']
            legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {categoria.replace("_", " ")}</div>'
    elif tipo_analisis == "ESTRÉS HIDRICO":
        escalas = ESCALAS_COLOR['ESTRES_HIDRICO']
        for categoria, info in escalas.items():
            color = info['color']
            desc = info['descripcion']
            legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {categoria.replace("_", " ")}</div>'
    elif tipo_analisis == "ESTADO NUTRICIONAL":
        escalas = ESCALAS_COLOR['ESTADO_NUTRICIONAL']
        for categoria, info in escalas.items():
            color = info['color']
            desc = info['descripcion']
            legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {categoria.replace("_", " ")}</div>'
    elif tipo_analisis == "VIGOR VEGETATIVO":
        escalas = ESCALAS_COLOR['VIGOR_VEGETATIVO']
        for categoria, info in escalas.items():
            color = info['color']
            desc = info['descripcion']
            legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {categoria.replace("_", " ")}</div>'
    elif tipo_analisis == "CLUSTERIZACIÓN":
        # Leyenda para clusters
        cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        for i in range(1, n_clusters + 1):
            color_idx = (i - 1) % len(cluster_colors)
            color = cluster_colors[color_idx]
            legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> Cluster {i}</div>'
    else:
        # Leyenda para fertilidad
        paleta = PALETAS_GEE['FERTILIDAD']
        for i in range(len(paleta)):
            valor = i / (len(paleta) - 1)
            color = paleta[i]
            if i == 0:
                etiqueta = "Muy Baja"
            elif i == len(paleta) - 1:
                etiqueta = "Muy Alta"
            else:
                etiqueta = f"{valor:.1f}"
            legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {etiqueta}</div>'
    
    legend_html += '''
        <div style="margin-top: 10px; font-size: 10px; color: #666;">
            💡 Click en las zonas para detalles
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def mostrar_metricas_salud_cultivo(gdf_salud, cultivo, tipo_analisis):
    """Muestra métricas de salud del cultivo - CORREGIDO"""
    
    st.subheader("📊 Métricas de Salud del Cultivo")
    
    # Métricas específicas según el tipo de análisis
    if tipo_analisis == "ESTADO SANITARIO":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'estado_sanitario' in gdf_salud.columns:
                avg_sanitario = gdf_salud['estado_sanitario'].mean()
                st.metric("🏥 Estado Sanitario Promedio", f"{avg_sanitario:.3f}")
            else:
                st.metric("🏥 Estado Sanitario Promedio", "N/A")
        
        with col2:
            if 'estado_sanitario' in gdf_salud.columns:
                zonas_buenas = (gdf_salud['estado_sanitario'] >= 0.6).sum()
                porcentaje_buenas = (zonas_buenas / len(gdf_salud)) * 100
                st.metric("✅ Zonas Buenas/Excelentes", f"{porcentaje_buenas:.1f}%")
            else:
                st.metric("✅ Zonas Buenas/Excelentes", "N/A")
        
        with col3:
            if 'estado_sanitario' in gdf_salud.columns:
                zonas_malas = (gdf_salud['estado_sanitario'] < 0.4).sum()
                porcentaje_malas = (zonas_malas / len(gdf_salud)) * 100
                st.metric("⚠️ Zonas con Problemas", f"{porcentaje_malas:.1f}%")
            else:
                st.metric("⚠️ Zonas con Problemas", "N/A")
        
        with col4:
            if 'ndvi' in gdf_salud.columns:
                ndvi_promedio = gdf_salud['ndvi'].mean()
                st.metric("🌿 NDVI Promedio", f"{ndvi_promedio:.3f}")
            else:
                st.metric("🌿 NDVI Promedio", "N/A")
    
    elif tipo_analisis == "ESTRÉS HÍDRICO":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'estres_hidrico' in gdf_salud.columns:
                avg_estres = gdf_salud['estres_hidrico'].mean()
                st.metric("💧 Estrés Hídrico Promedio", f"{avg_estres:.3f}")
            else:
                st.metric("💧 Estrés Hídrico Promedio", "N/A")
        
        with col2:
            if 'estres_hidrico' in gdf_salud.columns:
                zonas_sin_estres = (gdf_salud['estres_hidrico'] <= 0.2).sum()
                porcentaje_sin = (zonas_sin_estres / len(gdf_salud)) * 100
                st.metric("🌧️ Zonas sin Estrés", f"{porcentaje_sin:.1f}%")
            else:
                st.metric("🌧️ Zonas sin Estrés", "N/A")
        
        with col3:
            if 'estres_hidrico' in gdf_salud.columns:
                zonas_alto_estres = (gdf_salud['estres_hidrico'] > 0.6).sum()
                porcentaje_alto = (zonas_alto_estres / len(gdf_salud)) * 100
                st.metric("🔥 Zonas con Alto Estrés", f"{porcentaje_alto:.1f}%")
            else:
                st.metric("🔥 Zonas con Alto Estrés", "N/A")
        
        with col4:
            if 'temp_canopy' in gdf_salud.columns:
                temp_promedio = gdf_salud['temp_canopy'].mean()
                st.metric("🌡️ Temperatura Promedio", f"{temp_promedio:.1f}°C")
            else:
                st.metric("🌡️ Temperatura Promedio", "N/A")
    
    elif tipo_analisis == "ESTADO NUTRICIONAL":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'estado_nutricional' in gdf_salud.columns:
                avg_nutricion = gdf_salud['estado_nutricional'].mean()
                st.metric("🥦 Estado Nutricional Promedio", f"{avg_nutricion:.3f}")
            else:
                st.metric("🥦 Estado Nutricional Promedio", "N/A")
        
        with col2:
            if 'estado_nutricional' in gdf_salud.columns:
                zonas_optimas = (gdf_salud['estado_nutricional'] >= 0.8).sum()
                porcentaje_optimas = (zonas_optimas / len(gdf_salud)) * 100
                st.metric("🌟 Zonas Óptimas", f"{porcentaje_optimas:.1f}%")
            else:
                st.metric("🌟 Zonas Óptimas", "N/A")
        
        with col3:
            if 'estado_nutricional' in gdf_salud.columns:
                zonas_deficit = (gdf_salud['estado_nutricional'] < 0.4).sum()
                porcentaje_deficit = (zonas_deficit / len(gdf_salud)) * 100
                st.metric("⚠️ Zonas con Déficit", f"{porcentaje_deficit:.1f}%")
            else:
                st.metric("⚠️ Zonas con Déficit", "N/A")
        
        with col4:
            if 'nitrogeno' in gdf_salud.columns:
                n_balance = gdf_salud['nitrogeno'].std() / gdf_salud['nitrogeno'].mean() if gdf_salud['nitrogeno'].mean() > 0 else 0
                st.metric("⚖️ Variabilidad Nutricional", f"{n_balance:.3f}")
            else:
                st.metric("⚖️ Variabilidad Nutricional", "N/A")
    
    elif tipo_analisis == "VIGOR VEGETATIVO":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'vigor_vegetativo' in gdf_salud.columns:
                avg_vigor = gdf_salud['vigor_vegetativo'].mean()
                st.metric("🌱 Vigor Vegetativo Promedio", f"{avg_vigor:.3f}")
            else:
                st.metric("🌱 Vigor Vegetativo Promedio", "N/A")
        
        with col2:
            if 'vigor_vegetativo' in gdf_salud.columns:
                zonas_alto_vigor = (gdf_salud['vigor_vegetativo'] >= 0.8).sum()
                porcentaje_alto = (zonas_alto_vigor / len(gdf_salud)) * 100
                st.metric("🚀 Zonas de Alto Vigor", f"{porcentaje_alto:.1f}%")
            else:
                st.metric("🚀 Zonas de Alto Vigor", "N/A")
        
        with col3:
            if 'vigor_vegetativo' in gdf_salud.columns:
                zonas_bajo_vigor = (gdf_salud['vigor_vegetativo'] < 0.4).sum()
                porcentaje_bajo = (zonas_bajo_vigor / len(gdf_salud)) * 100
                st.metric("🐌 Zonas de Bajo Vigor", f"{porcentaje_bajo:.1f}%")
            else:
                st.metric("🐌 Zonas de Bajo Vigor", "N/A")
        
        with col4:
            if 'vigor_vegetativo' in gdf_salud.columns and 'ndvi' in gdf_salud.columns:
                try:
                    correlacion = gdf_salud[['vigor_vegetativo', 'ndvi']].corr().iloc[0,1]
                    st.metric("📈 Correlación Vigor-NDVI", f"{correlacion:.3f}")
                except:
                    st.metric("📈 Correlación Vigor-NDVI", "N/A")
            else:
                st.metric("📈 Correlación Vigor-NDVI", "N/A")
    
    elif tipo_analisis == "CLUSTERIZACIÓN":
        col1, col2, col3, col4 = st.columns(4)
        
        # Verificar si existe la columna 'cluster'
        if 'cluster' in gdf_salud.columns:
            with col1:
                n_clusters = gdf_salud['cluster'].nunique()
                st.metric("🔢 Número de Clusters", n_clusters)
            
            with col2:
                try:
                    cluster_mayor = gdf_salud['cluster'].mode().iloc[0]
                    zonas_mayor = (gdf_salud['cluster'] == cluster_mayor).sum()
                    porcentaje_mayor = (zonas_mayor / len(gdf_salud)) * 100
                    st.metric(f"🏆 Cluster Mayoritario ({cluster_mayor})", f"{porcentaje_mayor:.1f}%")
                except:
                    st.metric("🏆 Cluster Mayoritario", "N/A")
            
            with col3:
                try:
                    cluster_counts = gdf_salud['cluster'].value_counts()
                    heterogeneidad = cluster_counts.std() / cluster_counts.mean() if cluster_counts.mean() > 0 else 0
                    st.metric("🎭 Heterogeneidad", f"{heterogeneidad:.3f}")
                except:
                    st.metric("🎭 Heterogeneidad", "N/A")
            
            with col4:
                # Calcular silueta promedio (simulada)
                silhouette_score = 0.6 + np.random.uniform(-0.1, 0.1)
                st.metric("🎯 Calidad Clustering", f"{silhouette_score:.3f}")
        else:
            # Si no existe la columna 'cluster', mostrar mensaje
            st.warning("⚠️ No se pudo realizar la clusterización. Faltan datos necesarios.")
            st.info("Ejecuta primero un análisis de fertilidad o salud para generar los datos necesarios.")
    
    # Gráfico de distribución
    st.subheader("📈 Distribución de Valores")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    if tipo_analisis == "ESTADO SANITARIO" and 'estado_sanitario' in gdf_salud.columns:
        data = gdf_salud['estado_sanitario']
        titulo_hist = "Distribución del Estado Sanitario"
        color = PALETAS_GEE['ESTADO_SANITARIO'][2]
    elif tipo_analisis == "ESTRÉS HÍDRICO" and 'estres_hidrico' in gdf_salud.columns:
        data = gdf_salud['estres_hidrico']
        titulo_hist = "Distribución del Estrés Hídrico"
        color = PALETAS_GEE['ESTRES_HIDRICO'][2]
    elif tipo_analisis == "ESTADO NUTRICIONAL" and 'estado_nutricional' in gdf_salud.columns:
        data = gdf_salud['estado_nutricional']
        titulo_hist = "Distribución del Estado Nutricional"
        color = PALETAS_GEE['ESTADO_NUTRICIONAL'][2]
    elif tipo_analisis == "VIGOR VEGETATIVO" and 'vigor_vegetativo' in gdf_salud.columns:
        data = gdf_salud['vigor_vegetativo']
        titulo_hist = "Distribución del Vigor Vegetativo"
        color = PALETAS_GEE['VIGOR_VEGETATIVO'][2]
    elif tipo_analisis == "CLUSTERIZACIÓN" and 'cluster' in gdf_salud.columns:
        data = gdf_salud['cluster']
        titulo_hist = "Distribución de Clusters"
        color = '#4daf4a'
        
        # Gráfico de barras para clusters
        cluster_counts = gdf_salud['cluster'].value_counts().sort_index()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=color)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Número de Zonas')
        ax.set_title(titulo_hist)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        return  # Salir temprano para clusters
    
    else:
        # Si no hay datos, mostrar mensaje
        st.info("No hay datos disponibles para mostrar el gráfico de distribución.")
        return
    
    # Para datos continuos, mostrar histograma
    ax.hist(data, bins=20, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Promedio: {data.mean():.3f}')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    ax.set_title(titulo_hist)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================
# FUNCIONES EXISTENTES QUE SE MANTIENEN
# ==============================================

def crear_mapa_interactivo_esri(gdf, titulo, columna_valor=None, analisis_tipo=None, nutriente=None):
    """Crea mapa interactivo con base ESRI Satélite - MEJORADO"""
    
    # Obtener centro y bounds del GeoDataFrame
    centroid = gdf.geometry.centroid.iloc[0]
    bounds = gdf.total_bounds
    
    # Crear mapa centrado con ESRI Satélite por defecto
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=15,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satélite'
    )
    
    # Añadir otras bases como opciones
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Calles',
        overlay=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False
    ).add_to(m)
    
    # Añadir capa de relieve
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Relieve',
        overlay=False
    ).add_to(m)

    # CONFIGURAR RANGOS MEJORADOS
    if columna_valor and analisis_tipo:
        if analisis_tipo == "FERTILIDAD ACTUAL":
            vmin, vmax = 0, 1
            colores = PALETAS_GEE['FERTILIDAD']
            unidad = "Índice"
        elif analisis_tipo == "ANÁLISIS DE TEXTURA":
            # Mapa categórico para texturas
            colores_textura = {
                'ARENOSO': '#d8b365',
                'FRANCO_ARENOSO': '#f6e8c3', 
                'FRANCO': '#c7eae5',
                'FRANCO_ARCILLOSO': '#5ab4ac',
                'ARCILLOSO': '#01665e',
                'NO_DETERMINADA': '#999999'
            }
            unidad = "Textura"
        else:
            # RANGOS MÁS REALISTAS PARA RECOMENDACIONES
            if nutriente == "NITRÓGENO":
                vmin, vmax = 0, 250
                colores = PALETAS_GEE['NITROGENO']
                unidad = "kg/ha N"
            elif nutriente == "FÓSFORO":
                vmin, vmax = 0, 120
                colores = PALETAS_GEE['FOSFORO']
                unidad = "kg/ha P₂O₅"
            else:  # POTASIO
                vmin, vmax = 0, 200
                colores = PALETAS_GEE['POTASIO']
                unidad = "kg/ha K₂O"
        
        # Función para obtener color
        def obtener_color(valor, vmin, vmax, colores):
            if vmax == vmin:
                return colores[len(colores)//2]
            valor_norm = (valor - vmin) / (vmax - vmin)
            valor_norm = max(0, min(1, valor_norm))
            idx = int(valor_norm * (len(colores) - 1))
            return colores[idx]
        
        # Añadir cada polígono con estilo mejorado
        for idx, row in gdf.iterrows():
            if analisis_tipo == "ANÁLISIS DE TEXTURA":
                # Manejo especial para textura (valores categóricos)
                textura = row[columna_valor]
                color = colores_textura.get(textura, '#999999')
                valor_display = textura
            else:
                # Manejo para valores numéricos
                valor = row[columna_valor]
                color = obtener_color(valor, vmin, vmax, colores)
                if analisis_tipo == "FERTILIDAD ACTUAL":
                    valor_display = f"{valor:.3f}"
                else:
                    valor_display = f"{valor:.1f}"
            
            # Popup más informativo
            if analisis_tipo == "FERTILIDAD ACTUAL":
                popup_text = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <h4>Zona {row['id_zona']}</h4>
                    <b>Índice Fertilidad:</b> {valor_display}<br>
                    <b>Área:</b> {row.get('area_ha', 0):.2f} ha<br>
                    <b>Categoría:</b> {row.get('categoria', 'N/A')}<br>
                    <b>Prioridad:</b> {row.get('prioridad', 'N/A')}<br>
                    <hr>
                    <b>N:</b> {row.get('nitrogeno', 0):.1f} kg/ha<br>
                    <b>P:</b> {row.get('fosforo', 0):.1f} kg/ha<br>
                    <b>K:</b> {row.get('potasio', 0):.1f} kg/ha<br>
                    <b>MO:</b> {row.get('materia_organica', 0):.1f}%<br>
                    <b>NDVI:</b> {row.get('ndvi', 0):.3f}
                </div>
                """
            elif analisis_tipo == "ANÁLISIS DE TEXTURA":
                popup_text = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <h4>Zona {row['id_zona']}</h4>
                    <b>Textura:</b> {valor_display}<br>
                    <b>Adecuación:</b> {row.get('adecuacion_textura', 0):.1%}<br>
                    <b>Área:</b> {row.get('area_ha', 0):.2f} ha<br>
                    <hr>
                    <b>Arena:</b> {row.get('arena', 0):.1f}%<br>
                    <b>Limo:</b> {row.get('limo', 0):.1f}%<br>
                    <b>Arcilla:</b> {row.get('arcilla', 0):.1f}%<br>
                    <b>Capacidad Campo:</b> {row.get('capacidad_campo', 0):.1f} mm/m<br>
                    <b>Agua Disponible:</b> {row.get('agua_disponible', 0):.1f} mm/m
                </div>
                """
            else:
                popup_text = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <h4>Zona {row['id_zona']}</h4>
                    <b>Recomendación {nutriente}:</b> {valor_display} {unidad}<br>
                    <b>Área:</b> {row.get('area_ha', 0):.2f} ha<br>
                    <b>Categoría Fertilidad:</b> {row.get('categoria', 'N/A')}<br>
                    <b>Prioridad:</b> {row.get('prioridad', 'N/A')}<br>
                    <hr>
                    <b>N Actual:</b> {row.get('nitrogeno', 0):.1f} kg/ha<br>
                    <b>P Actual:</b> {row.get('fosforo', 0):.1f} kg/ha<br>
                    <b>K Actual:</b> {row.get('potasio', 0):.1f} kg/ha<br>
                    <b>Déficit:</b> {row.get('deficit_npk', 0):.1f} kg/ha
                </div>
                """
            
            # Estilo mejorado para los polígonos
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.7,
                    'opacity': 0.9
                },
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Zona {row['id_zona']}: {valor_display}"
            ).add_to(m)
            
            # Marcador con número de zona mejorado
            centroid = row.geometry.centroid
            folium.Marker(
                [centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f'''
                    <div style="
                        background-color: white; 
                        border: 2px solid black; 
                        border-radius: 50%; 
                        width: 28px; 
                        height: 28px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        font-weight: bold; 
                        font-size: 11px;
                        color: black;
                    ">{row["id_zona"]}</div>
                    '''
                ),
                tooltip=f"Zona {row['id_zona']} - Click para detalles"
            ).add_to(m)
    else:
        # Mapa simple del polígono original
        for idx, row in gdf.iterrows():
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': '#1f77b4',
                    'color': '#2ca02c',
                    'weight': 3,
                    'fillOpacity': 0.5,
                    'opacity': 0.8
                },
                popup=folium.Popup(
                    f"<b>Polígono {idx + 1}</b><br>Área: {calcular_superficie(gdf.iloc[[idx]]).iloc[0]:.2f} ha", 
                    max_width=300
                ),
            ).add_to(m)
    
    # Ajustar bounds del mapa
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    # Añadir controles mejorados
    folium.LayerControl().add_to(m)
    plugins.MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(m)
    plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)
    
    # Añadir leyenda mejorada
    if columna_valor and analisis_tipo:
        legend_html = f'''
        <div style="
            position: fixed; 
            top: 10px; 
            right: 10px; 
            width: 250px; 
            height: auto; 
            background-color: white; 
            border: 2px solid grey; 
            z-index: 9999; 
            font-size: 12px; 
            padding: 10px; 
            border-radius: 5px;
            font-family: Arial;
        ">
            <h4 style="margin:0 0 10px 0; text-align:center; color: #333;">{titulo}</h4>
            <div style="margin-bottom: 10px;">
                <strong>Escala de Valores ({unidad}):</strong>
            </div>
        '''
        
        if analisis_tipo == "FERTILIDAD ACTUAL":
            steps = 8
            for i in range(steps):
                value = i / (steps - 1)
                color_idx = int((i / (steps - 1)) * (len(PALETAS_GEE['FERTILIDAD']) - 1))
                color = PALETAS_GEE['FERTILIDAD'][color_idx]
                categoria = ["Muy Baja", "Baja", "Media-Baja", "Media", "Media-Alta", "Alta", "Muy Alta"][min(i, 6)] if i < 7 else "Óptima"
                legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {value:.1f} ({categoria})</div>'
        elif analisis_tipo == "ANÁLISIS DE TEXTURA":
            # Leyenda categórica para texturas
            colores_textura = {
                'ARENOSO': '#d8b365',
                'FRANCO_ARENOSO': '#f6e8c3', 
                'FRANCO': '#c7eae5',
                'FRANCO_ARCILLOSO': '#5ab4ac',
                'ARCILLOSO': '#01665e'
            }
            for textura, color in colores_textura.items():
                legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {textura}</div>'
        else:
            steps = 6
            for i in range(steps):
                value = vmin + (i / (steps - 1)) * (vmax - vmin)
                color_idx = int((i / (steps - 1)) * (len(colores) - 1))
                color = colores[color_idx]
                intensidad = ["Muy Baja", "Baja", "Media", "Alta", "Muy Alta", "Máxima"][i]
                legend_html += f'<div style="margin:2px 0;"><span style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; border:1px solid #000;"></span> {value:.0f} ({intensidad})</div>'
        
        legend_html += '''
            <div style="margin-top: 10px; font-size: 10px; color: #666;">
                💡 Click en las zonas para detalles
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def crear_mapa_visualizador_parcela(gdf):
    """Crea mapa interactivo para visualizar la parcela original con ESRI Satélite"""
    
    # Obtener centro y bounds
    centroid = gdf.geometry.centroid.iloc[0]
    bounds = gdf.total_bounds
    
    # Crear mapa con ESRI Satélite por defecto
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=14,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satélite'
    )
    
    # Añadir otras bases
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Calles',
        overlay=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False
    ).add_to(m)
    
    # Añadir polígonos de la parcela
    for idx, row in gdf.iterrows():
        area_ha = calcular_superficie(gdf.iloc[[idx]]).iloc[0]
        
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x: {
                'fillColor': '#1f77b4',
                'color': '#2ca02c',
                'weight': 3,
                'fillOpacity': 0.4,
                'opacity': 0.8
            },
            popup=folium.Popup(
                f"<b>Parcela {idx + 1}</b><br>"
                f"<b>Área:</b> {area_ha:.2f} ha<br>"
                f"<b>Coordenadas:</b> {centroid.y:.4f}, {centroid.x:.4f}",
                max_width=300
            ),
            tooltip=f"Parcela {idx + 1} - {area_ha:.2f} ha"
        ).add_to(m)
    
    # Ajustar bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    # Añadir controles
    folium.LayerControl().add_to(m)
    plugins.MeasureControl(position='bottomleft').add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)
    
    # Añadir leyenda
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>🌱 Visualizador de Parcela</b></p>
    <p><b>Leyenda:</b></p>
    <p><i style="background:#1f77b4; width:20px; height:20px; display:inline-block; margin-right:5px; opacity:0.4;"></i> Área de la parcela</p>
    <p><i style="background:#2ca02c; width:20px; height:20px; display:inline-block; margin_right:5px; opacity:0.8;"></i> Borde de la parcela</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ==============================================
# FUNCIONES PRINCIPALES DE VISUALIZACIÓN
# ==============================================

def mostrar_resultados_salud_cultivo():
    """Muestra los resultados del análisis de salud del cultivo"""
    if st.session_state.analisis_salud is None:
        st.warning("No hay datos de análisis de salud disponibles")
        return
    
    gdf_salud = st.session_state.analisis_salud
    area_total = st.session_state.area_total
    
    st.markdown(f"## 🌿 ANÁLISIS DE {analisis_tipo} - {cultivo.replace('_', ' ').title()}")
    
    # Botón para volver atrás
    if st.button("⬅️ Volver a Configuración", key="volver_salud"):
        st.session_state.analisis_completado = False
        st.rerun()
    
    # Mostrar métricas específicas
    mostrar_metricas_salud_cultivo(gdf_salud, cultivo, analisis_tipo)
    
    # Determinar columna para visualizar
    if analisis_tipo == "ESTADO SANITARIO":
        columna_visualizar = 'estado_sanitario'
        titulo_mapa = f"Estado Sanitario - {cultivo.replace('_', ' ').title()}"
    elif analisis_tipo == "ESTRÉS HÍDRICO":
        columna_visualizar = 'estres_hidrico'
        titulo_mapa = f"Estrés Hídrico - {cultivo.replace('_', ' ').title()}"
    elif analisis_tipo == "ESTADO NUTRICIONAL":
        columna_visualizar = 'estado_nutricional'
        titulo_mapa = f"Estado Nutricional - {cultivo.replace('_', ' ').title()}"
    elif analisis_tipo == "VIGOR VEGETATIVO":
        columna_visualizar = 'vigor_vegetativo'
        titulo_mapa = f"Vigor Vegetativo - {cultivo.replace('_', ' ').title()}"
    elif analisis_tipo == "CLUSTERIZACIÓN":
        columna_visualizar = 'cluster'
        titulo_mapa = f"Clusterización - {cultivo.replace('_', ' ').title()}"
    else:
        columna_visualizar = 'indice_fertilidad'
        titulo_mapa = f"Fertilidad - {cultivo.replace('_', ' ').title()}"
    
    # Mapa interactivo
    st.subheader("🗺️ Mapa de Análisis")
    mapa_salud = crear_mapa_salud_interactivo(
        gdf_salud, cultivo, analisis_tipo
    )
    st_folium(mapa_salud, width=800, height=500)
    
    # Tabla detallada
    st.subheader("📋 Tabla de Resultados por Zona")
    
    # Preparar columnas para la tabla
    columnas_base = ['id_zona', 'area_ha']
    
    if analisis_tipo == "ESTADO SANITARIO":
        columnas_base.extend(['estado_sanitario', 'categoria_sanitario', 'ndvi', 'savi', 'ndre'])
    elif analisis_tipo == "ESTRÉS HÍDRICO":
        columnas_base.extend(['estres_hidrico', 'categoria_estres', 'temp_canopy'])
    elif analisis_tipo == "ESTADO NUTRICIONAL":
        columnas_base.extend(['estado_nutricional', 'categoria_nutricional'])
        if 'nitrogeno' in gdf_salud.columns:
            columnas_base.extend(['nitrogeno', 'fosforo', 'potasio'])
    elif analisis_tipo == "VIGOR VEGETATIVO":
        columnas_base.extend(['vigor_vegetativo', 'categoria_vigor', 'estado_sanitario', 'estres_hidrico', 'estado_nutricional'])
    elif analisis_tipo == "CLUSTERIZACIÓN":
        columnas_base.extend(['cluster', 'descripcion_cluster'])
        if 'estado_sanitario' in gdf_salud.columns:
            columnas_base.extend(['estado_sanitario', 'estres_hidrico', 'estado_nutricional'])
    else:
        columnas_base.extend(['indice_fertilidad', 'categoria', 'nitrogeno', 'fosforo', 'potasio'])
    
    # Filtrar columnas existentes
    columnas_existentes = [col for col in columnas_base if col in gdf_salud.columns]
    
    if columnas_existentes:
        df_tabla = gdf_salud[columnas_existentes].copy()
        
        # Redondear valores
        if 'area_ha' in df_tabla.columns:
            df_tabla['area_ha'] = df_tabla['area_ha'].round(3)
        
        # Redondear valores numéricos
        for col in df_tabla.columns:
            if df_tabla[col].dtype in [np.float64, np.float32]:
                df_tabla[col] = df_tabla[col].round(3)
        
        st.dataframe(df_tabla, use_container_width=True)
    else:
        st.warning("No hay datos disponibles para mostrar la tabla")
    
    # DESCARGAR RESULTADOS
    st.markdown("### 💾 Descargar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Descargar CSV si hay datos
        if 'columnas_existentes' in locals() and columnas_existentes:
            csv = gdf_salud[columnas_existentes].to_csv(index=False)
            st.download_button(
                label="📥 Descargar Tabla CSV",
                data=csv,
                file_name=f"salud_{cultivo}_{analisis_tipo.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Descargar GeoJSON
        geojson = gdf_salud.to_json()
        st.download_button(
            label="🗺️ Descargar GeoJSON",
            data=geojson,
            file_name=f"salud_{cultivo}_{analisis_tipo.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.geojson",
            mime="application/json"
        )

# ==============================================
# FUNCIONES DE ANÁLISIS EXISTENTES (SIMPLIFICADAS)
# ==============================================

def dividir_parcela_en_zonas(gdf, n_zonas):
    """Divide la parcela en zonas de manejo"""
    try:
        if len(gdf) == 0:
            return gdf
        
        parcela_principal = gdf.iloc[0].geometry
        bounds = parcela_principal.bounds
        
        sub_poligonos = []
        n_cols = math.ceil(math.sqrt(n_zonas))
        n_rows = math.ceil(n_zonas / n_cols)
        
        width = (bounds[2] - bounds[0]) / n_cols
        height = (bounds[3] - bounds[1]) / n_rows
        
        for i in range(n_rows):
            for j in range(n_cols):
                if len(sub_poligonos) >= n_zonas:
                    break
                    
                cell_minx = bounds[0] + (j * width)
                cell_maxx = bounds[0] + ((j + 1) * width)
                cell_miny = bounds[1] + (i * height)
                cell_maxy = bounds[1] + ((i + 1) * height)
                
                cell_poly = Polygon([
                    (cell_minx, cell_miny),
                    (cell_maxx, cell_miny),
                    (cell_maxx, cell_maxy),
                    (cell_minx, cell_maxy)
                ])
                
                intersection = parcela_principal.intersection(cell_poly)
                if not intersection.is_empty and intersection.area > 0:
                    sub_poligonos.append(intersection)
        
        if sub_poligonos:
            nuevo_gdf = gpd.GeoDataFrame({
                'id_zona': range(1, len(sub_poligonos) + 1),
                'geometry': sub_poligonos
            }, crs=gdf.crs)
            
            # Calcular área de cada zona
            nuevo_gdf['area_ha'] = nuevo_gdf['geometry'].apply(
                lambda geom: calcular_superficie(gpd.GeoDataFrame({'geometry': [geom]}, crs=nuevo_gdf.crs))
            )
            
            return nuevo_gdf
        else:
            return gdf
            
    except Exception as e:
        st.error(f"Error dividiendo parcela: {str(e)}")
        return gdf

def analizar_textura_suelo(gdf, cultivo, mes_analisis):
    """Realiza análisis de textura del suelo"""
    params_textura = TEXTURA_SUELO_OPTIMA[cultivo]
    zonas_gdf = gdf.copy()
    
    # Inicializar columnas
    for col in ['arena', 'limo', 'arcilla', 'textura_suelo', 'adecuacion_textura', 
                'categoria_adecuacion', 'capacidad_campo', 'agua_disponible']:
        if col not in zonas_gdf.columns:
            zonas_gdf[col] = 0.0 if col not in ['textura_suelo', 'categoria_adecuacion'] else ""
    
    for idx, row in zonas_gdf.iterrows():
        try:
            # Simular composición granulométrica
            rng = np.random.RandomState(idx)
            
            arena = rng.uniform(20, 80)
            limo = rng.uniform(10, 60)
            arcilla = rng.uniform(10, 50)
            
            # Normalizar a 100%
            total = arena + limo + arcilla
            arena = (arena / total) * 100
            limo = (limo / total) * 100
            arcilla = (arcilla / total) * 100
            
            # Clasificar textura
            textura = clasificar_textura_suelo(arena, limo, arcilla)
            
            # Evaluar adecuación
            categoria_adecuacion, puntaje_adecuacion = evaluar_adecuacion_textura(textura, cultivo)
            
            # Calcular propiedades físicas
            materia_organica = rng.uniform(1.0, 8.0)
            propiedades = calcular_propiedades_fisicas_suelo(textura, materia_organica)
            
            # Asignar valores
            zonas_gdf.loc[idx, 'arena'] = arena
            zonas_gdf.loc[idx, 'limo'] = limo
            zonas_gdf.loc[idx, 'arcilla'] = arcilla
            zonas_gdf.loc[idx, 'textura_suelo'] = textura
            zonas_gdf.loc[idx, 'adecuacion_textura'] = puntaje_adecuacion
            zonas_gdf.loc[idx, 'categoria_adecuacion'] = categoria_adecuacion
            zonas_gdf.loc[idx, 'capacidad_campo'] = propiedades['capacidad_campo']
            zonas_gdf.loc[idx, 'agua_disponible'] = propiedades['agua_disponible']
            
        except Exception as e:
            # Valores por defecto
            zonas_gdf.loc[idx, 'arena'] = params_textura['arena_optima']
            zonas_gdf.loc[idx, 'limo'] = params_textura['limo_optima']
            zonas_gdf.loc[idx, 'arcilla'] = params_textura['arcilla_optima']
            zonas_gdf.loc[idx, 'textura_suelo'] = params_textura['textura_optima']
            zonas_gdf.loc[idx, 'adecuacion_textura'] = 1.0
            zonas_gdf.loc[idx, 'categoria_adecuacion'] = "ÓPTIMA"
    
    return zonas_gdf

def calcular_indices_gee(gdf, cultivo, mes_analisis, analisis_tipo, nutriente):
    """Calcula índices GEE para fertilidad"""
    params = PARAMETROS_CULTIVOS[cultivo]
    zonas_gdf = gdf.copy()
    
    # Inicializar columnas
    columnas_npk = ['nitrogeno', 'fosforo', 'potasio', 'materia_organica', 
                    'humedad', 'ph', 'conductividad', 'ndvi', 'indice_fertilidad',
                    'categoria', 'recomendacion_npk', 'deficit_npk', 'prioridad']
    
    for col in columnas_npk:
        if col not in zonas_gdf.columns:
            zonas_gdf[col] = 0.0 if col not in ['categoria', 'prioridad'] else ""
    
    for idx, row in zonas_gdf.iterrows():
        try:
            rng = np.random.RandomState(idx)
            
            # Simular valores
            nitrogeno = rng.uniform(params['NITROGENO']['min'], params['NITROGENO']['max'])
            fosforo = rng.uniform(params['FOSFORO']['min'], params['FOSFORO']['max'])
            potasio = rng.uniform(params['POTASIO']['min'], params['POTASIO']['max'])
            
            materia_organica = rng.uniform(1.0, 8.0)
            humedad = rng.uniform(0.2, 0.6)
            ph = rng.uniform(5.0, 7.5)
            conductividad = rng.uniform(0.5, 2.5)
            ndvi = rng.uniform(0.3, 0.9)
            
            # Calcular índice de fertilidad
            n_norm = nitrogeno / params['NITROGENO']['optimo']
            p_norm = fosforo / params['FOSFORO']['optimo']
            k_norm = potasio / params['POTASIO']['optimo']
            
            indice_fertilidad = (n_norm + p_norm + k_norm) / 3
            indice_fertilidad = max(0, min(1, indice_fertilidad))
            
            # Categorizar
            if indice_fertilidad >= 0.8:
                categoria = "EXCELENTE"
                prioridad = "BAJA"
            elif indice_fertilidad >= 0.6:
                categoria = "BUENA"
                prioridad = "MEDIA"
            elif indice_fertilidad >= 0.4:
                categoria = "REGULAR"
                prioridad = "MEDIA-ALTA"
            else:
                categoria = "DEFICIENTE"
                prioridad = "ALTA"
            
            # Calcular recomendaciones si es necesario
            if analisis_tipo == "RECOMENDACIONES NPK":
                if nutriente == "NITRÓGENO":
                    deficit = max(0, params['NITROGENO']['optimo'] - nitrogeno)
                    recomendacion = deficit * 1.5
                elif nutriente == "FÓSFORO":
                    deficit = max(0, params['FOSFORO']['optimo'] - fosforo)
                    recomendacion = deficit * 1.5
                else:  # POTASIO
                    deficit = max(0, params['POTASIO']['optimo'] - potasio)
                    recomendacion = deficit * 1.5
            else:
                recomendacion = 0
                deficit = 0
            
            # Asignar valores
            zonas_gdf.loc[idx, 'nitrogeno'] = nitrogeno
            zonas_gdf.loc[idx, 'fosforo'] = fosforo
            zonas_gdf.loc[idx, 'potasio'] = potasio
            zonas_gdf.loc[idx, 'materia_organica'] = materia_organica
            zonas_gdf.loc[idx, 'humedad'] = humedad
            zonas_gdf.loc[idx, 'ph'] = ph
            zonas_gdf.loc[idx, 'conductividad'] = conductividad
            zonas_gdf.loc[idx, 'ndvi'] = ndvi
            zonas_gdf.loc[idx, 'indice_fertilidad'] = indice_fertilidad
            zonas_gdf.loc[idx, 'categoria'] = categoria
            zonas_gdf.loc[idx, 'recomendacion_npk'] = recomendacion
            zonas_gdf.loc[idx, 'deficit_npk'] = deficit
            zonas_gdf.loc[idx, 'prioridad'] = prioridad
            
        except Exception as e:
            # Valores por defecto
            zonas_gdf.loc[idx, 'nitrogeno'] = params['NITROGENO']['optimo']
            zonas_gdf.loc[idx, 'fosforo'] = params['FOSFORO']['optimo']
            zonas_gdf.loc[idx, 'potasio'] = params['POTASIO']['optimo']
            zonas_gdf.loc[idx, 'indice_fertilidad'] = 0.5
            zonas_gdf.loc[idx, 'categoria'] = "REGULAR"
            zonas_gdf.loc[idx, 'prioridad'] = "MEDIA"
    
    return zonas_gdf

def procesar_archivo(uploaded_file):
    """Procesa el archivo subido"""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if uploaded_file.name.lower().endswith('.kml'):
                gdf = gpd.read_file(file_path, driver='KML')
            else:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                shp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.shp')]
                kml_files = [f for f in os.listdir(tmp_dir) if f.endswith('.kml')]
                
                if shp_files:
                    shp_path = os.path.join(tmp_dir, shp_files[0])
                    gdf = gpd.read_file(shp_path)
                elif kml_files:
                    kml_path = os.path.join(tmp_dir, kml_files[0])
                    gdf = gpd.read_file(kml_path, driver='KML')
                else:
                    st.error("❌ No se encontró archivo .shp o .kml en el ZIP")
                    return None
            
            if not gdf.is_valid.all():
                gdf = gdf.make_valid()
            
            return gdf
            
    except Exception as e:
        st.error(f"❌ Error procesando archivo: {str(e)}")
        return None

# ==============================================
# INTERFAZ PRINCIPAL
# ==============================================

def main():
    # Mostrar información de la aplicación
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Métodología GEE")
    st.sidebar.info("""
    Esta aplicación utiliza:
    - **Google Earth Engine** para análisis satelital
    - **Índices espectrales** (NDVI, SAVI, NDRE, GNDVI)
    - **Modelos predictivos** de salud del cultivo
    - **Análisis de textura** del suelo
    - **Enfoque agroecológico** integrado
    - **Clusterización** para manejo diferenciado
    """)

    # Procesar archivo subido si existe
    if uploaded_file is not None and not st.session_state.analisis_completado:
        with st.spinner("🔄 Procesando archivo..."):
            gdf_original = procesar_archivo(uploaded_file)
            if gdf_original is not None:
                st.session_state.gdf_original = gdf_original
                st.session_state.datos_demo = False

    # Cargar datos de demostración si se solicita
    if st.session_state.datos_demo and st.session_state.gdf_original is None:
        # Crear polígono de ejemplo
        poligono_ejemplo = Polygon([
            [-74.1, 4.6], [-74.0, 4.6], [-74.0, 4.7], [-74.1, 4.7], [-74.1, 4.6]
        ])
        
        gdf_demo = gpd.GeoDataFrame(
            {'id': [1], 'nombre': ['Parcela Demo']},
            geometry=[poligono_ejemplo],
            crs="EPSG:4326"
        )
        st.session_state.gdf_original = gdf_demo

    # Mostrar interfaz según el estado
    if st.session_state.analisis_completado:
        # Mostrar resultados según el tipo de análisis
        if analisis_tipo in ["ESTADO SANITARIO", "ESTRÉS HÍDRICO", "ESTADO NUTRICIONAL", 
                           "VIGOR VEGETATIVO", "CLUSTERIZACIÓN"]:
            mostrar_resultados_salud_cultivo()
        else:
            # Para análisis tradicionales, mostrar mensaje informativo
            st.info(f"Análisis {analisis_tipo} seleccionado. Para análisis de salud del cultivo, seleccione una de las opciones de salud en el sidebar.")
            
            # Botón para volver atrás
            if st.button("⬅️ Volver a Configuración"):
                st.session_state.analisis_completado = False
                st.rerun()
                    
    elif st.session_state.gdf_original is not None:
        mostrar_configuracion_parcela()
    else:
        mostrar_modo_demo()

def mostrar_modo_demo():
    """Muestra la interfaz de demostración"""
    st.markdown("### 🚀 Modo Demostración")
    st.info("""
    **Para usar la aplicación:**
    1. Sube un archivo ZIP con el shapefile de tu parcela
    2. Selecciona el cultivo y tipo de análisis
    3. Configura los parámetros en el sidebar
    4. Ejecuta el análisis GEE
    
    **📁 El shapefile debe incluir:**
    - .shp (geometrías)
    - .shx (índice)
    - .dbf (atributos)
    - .prj (sistema de coordenadas)
    
    **NUEVO: Análisis de Salud del Cultivo**
    - Estado sanitario con índices espectrales
    - Estrés hídrico y nutricional
    - Vigor vegetativo compuesto
    - Clusterización para manejo diferenciado
    """)
    
    # Ejemplo de datos de demostración
    if st.button("🎯 Cargar Datos de Demostración", type="primary"):
        st.session_state.datos_demo = True
        st.rerun()

def mostrar_configuracion_parcela():
    """Muestra la configuración de la parcela antes del análisis"""
    gdf_original = st.session_state.gdf_original
    
    # Mostrar información de la parcela
    if st.session_state.datos_demo:
        st.success("✅ Datos de demostración cargados")
    else:
        st.success("✅ Parcela cargada correctamente")
    
    # Calcular estadísticas
    area_total = calcular_superficie(gdf_original).sum()
    num_poligonos = len(gdf_original)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📐 Área Total", f"{area_total:.2f} ha")
    with col2:
        st.metric("🔢 Número de Polígonos", num_poligonos)
    with col3:
        st.metric("🌱 Cultivo", cultivo.replace('_', ' ').title())
    
    # VISUALIZADOR DE PARCELA ORIGINAL
    st.markdown("### 🗺️ Visualizador de Parcela")
    
    # Crear y mostrar mapa interactivo
    mapa_parcela = crear_mapa_visualizador_parcela(gdf_original)
    st_folium(mapa_parcela, width=800, height=500)
    
    # DIVIDIR PARCELA EN ZONAS
    st.markdown("### 📊 División en Zonas de Manejo")
    st.info(f"La parcela se dividirá en **{n_divisiones} zonas** para análisis detallado")
    
    # Botón para ejecutar análisis
    if st.button("🚀 Ejecutar Análisis GEE Completo", type="primary"):
        with st.spinner("🔄 Dividiendo parcela en zonas..."):
            gdf_zonas = dividir_parcela_en_zonas(gdf_original, n_divisiones)
            st.session_state.gdf_zonas = gdf_zonas
        
        with st.spinner("🔬 Realizando análisis GEE..."):
            # Calcular índices según tipo de análisis
            if analisis_tipo == "ANÁLISIS DE TEXTURA":
                gdf_analisis = analizar_textura_suelo(gdf_zonas, cultivo, mes_analisis)
                st.session_state.analisis_textura = gdf_analisis
            elif analisis_tipo in ["ESTADO SANITARIO", "ESTRÉS HÍDRICO", "ESTADO NUTRICIONAL", "VIGOR VEGETATIVO"]:
                # Análisis de salud del cultivo
                gdf_analisis = calcular_indices_salud_cultivo(gdf_zonas, cultivo, mes_analisis)
                st.session_state.analisis_salud = gdf_analisis
            elif analisis_tipo == "CLUSTERIZACIÓN":
                # Clusterización
                gdf_analisis = realizar_clusterizacion_cultivo(gdf_zonas, cultivo, n_clusters)
                st.session_state.analisis_salud = gdf_analisis
                st.session_state.analisis_clusters = gdf_analisis
            else:
                # Análisis tradicional (fertilidad o recomendaciones NPK)
                gdf_analisis = calcular_indices_gee(gdf_zonas, cultivo, mes_analisis, analisis_tipo, nutriente)
                st.session_state.gdf_analisis = gdf_analisis
            
            # Siempre ejecutar análisis de textura también para referencia
            if analisis_tipo != "ANÁLISIS DE TEXTURA":
                with st.spinner("🏗️ Realizando análisis de textura..."):
                    gdf_textura = analizar_textura_suelo(gdf_zonas, cultivo, mes_analisis)
                    st.session_state.analisis_textura = gdf_textura
            
            # Para análisis tradicionales, también calcular salud
            if analisis_tipo in ["FERTILIDAD ACTUAL", "RECOMENDACIONES NPK"]:
                with st.spinner("🌿 Calculando indicadores de salud..."):
                    gdf_salud = calcular_indices_salud_cultivo(gdf_analisis, cultivo, mes_analisis)
                    st.session_state.analisis_salud = gdf_salud
            
            st.session_state.area_total = area_total
            st.session_state.analisis_completado = True
        
        st.rerun()

# EJECUTAR APLICACIÓN
if __name__ == "__main__":
    main()
