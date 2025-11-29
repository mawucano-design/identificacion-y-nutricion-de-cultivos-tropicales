import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict

class PalmLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_size)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Atención
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Último paso de tiempo
        last_output = attended_out[:, -1, :]
        
        # Salidas
        classification = self.classifier(last_output)
        regression = self.regressor(last_output)
        
        return classification, regression, attention_weights

def predict_stress(time_series_data: pd.DataFrame, sequence_length: int = 8):
    """
    Predice estrés basado en series temporales
    """
    # Preprocesar datos
    features = ['ndvi', 'gndvi', 'ndre', 'temp']  # Características a usar
    
    # Crear secuencias deslizantes
    sequences = []
    targets = []
    
    for palm_id in time_series_data['palm_id'].unique():
        palm_data = time_series_data[time_series_data['palm_id'] == palm_id]
        palm_data = palm_data.sort_values('date')
        
        values = palm_data[features].values
        
        # Crear secuencias
        for i in range(len(values) - sequence_length):
            sequence = values[i:i + sequence_length]
            target = values[i + sequence_length, 0]  # NDVI como target
            
            sequences.append(sequence)
            targets.append(target)
    
    # Convertir a tensores
    X = torch.FloatTensor(sequences)
    y = torch.FloatTensor(targets)
    
    return X, y

def calculate_stress_risk(ndvi_trend: List[float], current_ndvi: float) -> Dict:
    """
    Calcula el riesgo de estrés basado en tendencia y valor actual
    """
    if len(ndvi_trend) < 3:
        return {'risk_level': 'UNKNOWN', 'confidence': 0.0}
    
    # Calcular tendencia
    trend = np.polyfit(range(len(ndvi_trend)), ndvi_trend, 1)[0]
    
    # Evaluar riesgo
    if current_ndvi < 0.3:
        risk_level = 'CRITICAL'
        confidence = 0.95
    elif current_ndvi < 0.5:
        risk_level = 'HIGH'
        confidence = 0.8
    elif current_ndvi < 0.6 and trend < -0.02:
        risk_level = 'MEDIUM'
        confidence = 0.7
    elif trend < -0.01:
        risk_level = 'LOW'
        confidence = 0.6
    else:
        risk_level = 'LOW'
        confidence = 0.3
    
    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'trend': trend,
        'recommendation': generate_recommendation(risk_level, current_ndvi)
    }

def generate_recommendation(risk_level: str, current_ndvi: float) -> str:
    """
    Genera recomendaciones basadas en el nivel de riesgo
    """
    recommendations = {
        'CRITICAL': 'Intervención inmediata requerida. Revisar riego, nutrientes y posible enfermedad.',
        'HIGH': 'Aumentar frecuencia de monitoreo. Verificar condiciones del suelo y aplicar fertilizante si es necesario.',
        'MEDIUM': 'Monitorear estrechamente. Considerar ajustes menores en riego.',
        'LOW': 'Condiciones normales. Continuar monitoreo regular.'
    }
    
    return recommendations.get(risk_level, 'Continuar monitoreo regular.')
