import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st

class PalmDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Arquitectura simple para segmentación
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.conv3(x))
        return x

def segment_individual_palms(image_array, min_area=50):
    """
    Segmenta palmas individuales en una imagen
    """
    # Convertir a escala de grises
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Aplicar filtro Gaussian
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold adaptativo
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    palm_data = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area > min_area:
            # Calcular momento para el centroide
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                palm_data.append({
                    'id': f'P{i+1:03d}',
                    'x': cx,
                    'y': cy,
                    'area': area,
                    'contour': contour
                })
    
    return palm_data

def calculate_ndvi(red_band, nir_band):
    """
    Calcula el índice NDVI a partir de bandas roja e infrarroja
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    # Evitar división por cero
    denominator = nir + red
    denominator[denominator == 0] = 1e-10
    
    ndvi = (nir - red) / denominator
    return np.clip(ndvi, -1, 1)
