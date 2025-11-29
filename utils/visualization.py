import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

def create_ndvi_timeseries_plot(df: pd.DataFrame, palm_id: str) -> go.Figure:
    """
    Crea gráfico de series temporales de NDVI para una palma específica
    """
    palm_data = df[df['palm_id'] == palm_id].sort_values('date')
    
    fig = go.Figure()
    
    # Línea principal de NDVI
    fig.add_trace(go.Scatter(
        x=palm_data['date'],
        y=palm_data['ndvi'],
        mode='lines+markers',
        name='NDVI',
        line=dict(color='#2e7d32', width=3),
        marker=dict(size=8)
    ))
    
    # Bandas de referencia
    fig.add_hrect(y0=0.7, y1=1.0, 
                 fillcolor="green", opacity=0.1, 
                 annotation_text="Óptimo", annotation_position="right")
    
    fig.add_hrect(y0=0.6, y1=0.7, 
                 fillcolor="lightgreen", opacity=0.1,
                 annotation_text="Saludable", annotation_position="right")
    
    fig.add_hrect(y0=0.3, y1=0.6, 
                 fillcolor="orange", opacity=0.1,
                 annotation_text="Estrés Leve", annotation_position="right")
    
    fig.add_hrect(y0=0.0, y1=0.3, 
                 fillcolor="red", opacity=0.1,
                 annotation_text="Estrés Severo", annotation_position="right")
    
    fig.update_layout(
        title=f"Evolución de NDVI - {palm_id}",
        xaxis_title="Fecha",
        yaxis_title="NDVI",
        height=400,
        showlegend=True
    )
    
    return fig

def create_palm_distribution_map(palm_data: List[Dict], field_size: tuple = (1000, 1000)) -> go.Figure:
    """
    Crea mapa de distribución de palmas
    """
    if not palm_data:
        return go.Figure()
    
    df = pd.DataFrame(palm_data)
    
    fig = px.scatter(
        df, x='x', y='y', size='area',
        title="Distribución de Palmeras",
        hover_data=['id', 'area'],
        size_max=20
    )
    
    # Calcular densidad
    from scipy.spatial import Voronoi, voronoi_plot_2d
    try:
        points = df[['x', 'y']].values
        vor = Voronoi(points)
        
        # Agregar diagrama de Voronoi (opcional)
        for simplex in vor.ridge_vertices:
            if -1 not in simplex:
                fig.add_trace(go.Scatter(
                    x=[vor.vertices[simplex, 0][0], vor.vertices[simplex, 1][0]],
                    y=[vor.vertices[simplex, 0][1], vor.vertices[simplex, 1][1]],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False
                ))
    except:
        pass
    
    fig.update_layout(
        xaxis_title="Coordenada X (píxeles)",
        yaxis_title="Coordenada Y (píxeles)",
        height=500
    )
    
    return fig

def create_stress_heatmap(ndvi_matrix: np.ndarray) -> go.Figure:
    """
    Crea heatmap de estrés basado en matriz NDVI
    """
    fig = go.Figure(data=go.Heatmap(
        z=ndvi_matrix,
        colorscale=[
            [0, 'red'],
            [0.3, 'orange'],
            [0.6, 'yellow'],
            [0.7, 'lightgreen'],
            [1, 'green']
        ],
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title="Mapa de Calor - Distribución de NDVI",
        xaxis_title="Coordenada X",
        yaxis_title="Coordenada Y",
        height=500
    )
    
    return fig
