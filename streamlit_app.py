import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# -------------- Interface utilisateur --------------
st.title("📈 ECG Miner - Analyse AI des ECG 12 Dérivations")

# Sélection du format et layout ECG
st.subheader("📌 Sélectionnez le format et la disposition des dérivations ECG")

ecg_format = st.selectbox(
    "Format de l'ECG",
    ["1 page, 12x1", "2 pages, 6x1", "1 page, 6x2", "1 page, 6x2 avec dérivation rythmique", "1 page, 3x4"]
)

st.write(f"🖼️ Format sélectionné : **{ecg_format}**")

# Téléchargement de l'image ECG
uploaded_file = st.file_uploader("📸 Téléchargez une image de votre ECG (JPG, PNG)", type=["jpg", "jpeg", "png"])

# -------------- Traitement de l'image avec OpenCV --------------
def preprocess_ecg_image(image):
    """Améliore la qualité de l’image ECG sans perdre d’informations."""
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Égalisation d’histogramme pour améliorer le contraste
    enhanced = cv2.equalizeHist(gray)

    # Seuillage adaptatif pour mieux extraire les tracés ECG
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)
    return binary

# -------------- Extraction des dérivations ECG --------------
def extract_ecg_leads(image, ecg_format):
    """ Détecte et extrait les 12 dérivations en fonction du format ECG sélectionné. """
    height, width = image.shape
    leads = {}

    if ecg_format == "1 page, 12x1":
        lead_height = height // 12
        for i in range(12):
            leads[f"Lead {i+1}"] = image[i * lead_height:(i + 1) * lead_height, :]
    
    elif ecg_format == "2 pages, 6x1":
        lead_height = height // 6
        for i in range(6):
            leads[f"Lead {i+1}"] = image[i * lead_height:(i + 1) * lead_height, :]
        for i in range(6):
            leads[f"Lead {i+7}"] = image[i * lead_height:(i + 1) * lead_height, :]

    elif ecg_format == "1 page, 6x2":
        lead_height = height // 6
        mid_width = width // 2
        for i in range(6):
            leads[f"Lead {i+1}"] = image[i * lead_height:(i + 1) * lead_height, :mid_width]
            leads[f"Lead {i+7}"] = image[i * lead_height:(i + 1) * lead_height, mid_width:]

    elif ecg_format == "1 page, 6x2 avec dérivation rythmique":
        lead_height = height // 6
        mid_width = width // 2
        for i in range(6):
            leads[f"Lead {i+1}"] = image[i * lead_height:(i + 1) * lead_height, :mid_width]
            leads[f"Lead {i+7}"] = image[i * lead_height:(i + 1) * lead_height, mid_width:]
        leads["Lead Rythmique"] = image[5 * lead_height:(6) * lead_height, :]

    elif ecg_format == "1 page, 3x4":
        lead_height = height // 3
        mid_width = width // 4
        for i in range(3):
            for j in range(4):
                leads[f"Lead {i*4+j+1}"] = image[i * lead_height:(i + 1) * lead_height, j * mid_width:(j + 1) * mid_width]

    return leads

# -------------- Exécution du Code --------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Améliorer l’image avec OpenCV
    gray = preprocess_ecg_image(image)

    # Affichage de l’image traitée
    st.image(gray, caption="📷 Image ECG après traitement", use_column_width=True)

    # Extraire les dérivations
    leads = extract_ecg_leads(gray, ecg_format)

    # Affichage des dérivations
    fig, axes = plt.subplots(len(leads) // 2, 2, figsize=(12, 8))
    for i, (lead, signal) in enumerate(leads.items()):
        ax = axes[i // 2, i % 2]
        ax.imshow(signal, cmap="gray", aspect="auto")
        ax.set_title(lead)
        ax.axis("off")

    st.pyplot(fig)