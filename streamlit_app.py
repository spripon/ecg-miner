import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Charger le modèle de digitalisation ECG (à entraîner et ajouter au repo)
MODEL_PATH = "ecg_digitization_model.pth"

# -------------- Interface utilisateur --------------
st.title("📈 ECG Interpretation - Digitalisation et Analyse IA")

# Téléchargement de l'image ECG
uploaded_file = st.file_uploader("📸 Téléchargez une image de votre ECG (JPG, PNG)", type=["jpg", "jpeg", "png"])

# -------------- Prétraitement de l’image avec OpenCV --------------
def preprocess_ecg_image(image):
    """Améliore la qualité de l’image ECG et prépare la digitalisation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Équalisation d’histogramme pour améliorer le contraste
    enhanced = cv2.equalizeHist(gray)

    # Suppression des grilles et artefacts
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)
    return binary

# -------------- Extraction des Dérivations ECG --------------
def extract_ecg_leads(image):
    """Détecte et segmente les dérivations ECG sur l'image."""
    height, width = image.shape
    lead_height = height // 12  # Supposons 12 dérivations en colonnes

    leads = {}
    for i in range(12):
        leads[f"Lead {i+1}"] = image[i * lead_height:(i + 1) * lead_height, :]

    return leads

# -------------- Digitalisation des Tracés ECG avec IA --------------
def digitize_ecg_signal(image):
    """Convertit les tracés ECG en signaux numériques grâce à un modèle IA."""
    if not MODEL_PATH:
        st.warning("⚠️ Modèle IA de digitalisation non disponible. Chargez un modèle entraîné.")
        return None

    # Charger l’image sous forme de tenseur
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = Image.fromarray(image)
    img_tensor = transform(img).unsqueeze(0)

    # Charger le modèle (si disponible)
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
    
    # Convertir la sortie du modèle en signal ECG (simulé ici)
    signal = output.numpy().flatten()

    return signal

# -------------- Exécution du Code --------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Améliorer l’image avec OpenCV
    processed_image = preprocess_ecg_image(image)

    # Affichage de l’image traitée
    st.image(processed_image, caption="📷 Image ECG après traitement", use_column_width=True)

    # Extraire les dérivations
    leads = extract_ecg_leads(processed_image)

    # Affichage des dérivations
    fig, axes = plt.subplots(6, 2, figsize=(12, 8))
    for i, (lead, signal) in enumerate(leads.items()):
        ax = axes[i // 2, i % 2]
        ax.imshow(signal, cmap="gray", aspect="auto")
        ax.set_title(lead)
        ax.axis("off")

    st.pyplot(fig)

    # Digitalisation IA
    st.subheader("📊 Digitalisation des Tracés ECG")
    lead_selected = st.selectbox("Sélectionnez une dérivation à digitaliser", list(leads.keys()))
    digitized_signal = digitize_ecg_signal(leads[lead_selected])

    if digitized_signal is not None:
        # Affichage du signal numérique reconstruit
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(digitized_signal, color="red")
        ax.set_title(f"Signal ECG digitalisé - {lead_selected}")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)