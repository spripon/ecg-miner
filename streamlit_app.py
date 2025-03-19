import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    """Applique un traitement OpenCV pour améliorer l’extraction des dérivations ECG."""
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre Gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours avec un seuillage adaptatif
    edges = cv2.Canny(blurred, 50, 150)

    # Détection des lignes avec Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Création d’un masque pour aligner et nettoyer l'image
    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

    # Inversion du masque pour améliorer la segmentation
    final_image = cv2.bitwise_and(gray, mask)

    return final_image

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

# -------------- Animation du Tracé ECG --------------
def animate_ecg(signal):
    """Anime le tracé ECG pour simuler un signal dynamique."""
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    line, = ax.plot([], [], "r-", animated=True)

    def init():
        ax.set_xlim(0, len(signal))
        ax.set_ylim(np.min(signal) - 0.1, np.max(signal) + 0.1)
        return line,

    def update(frame):
        x_data.append(frame)
        y_data.append(signal[frame])
        line.set_data(x_data, y_data)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(signal), init_func=init, blit=True, interval=10)

    st.pyplot(fig)

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

    # Animation ECG
    st.subheader("📊 Animation du Tracé ECG")
    lead_selected = st.selectbox("Sélectionnez une dérivation à animer", list(leads.keys()))
    animate_ecg(leads[lead_selected])