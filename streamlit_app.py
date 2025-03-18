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

# -------------- Traitement de l'image --------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Affichage de l’image téléchargée
    st.image(gray, caption="📷 Image ECG téléchargée", use_column_width=True)

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

    # Extraction des signaux ECG
    leads = extract_ecg_leads(gray, ecg_format)

    # -------------- Affichage des dérivations --------------
    fig, axes = plt.subplots(len(leads) // 2, 2, figsize=(12, 8))

    for i, (lead, signal) in enumerate(leads.items()):
        ax = axes[i // 2, i % 2]
        ax.imshow(signal, cmap="gray", aspect="auto")
        ax.set_title(lead)
        ax.axis("off")

    st.pyplot(fig)

    # -------------- Prédiction AI --------------
    def predict_ecg_condition(image):
        """ Prédiction AI pour interpréter l’ECG. """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        img = Image.fromarray(image)
        img_tensor = transform(img).unsqueeze(0)

        # Charger le modèle AI entraîné (uploadé sur GitHub)
        model = torch.load("ecg_model.pth", map_location=torch.device('cpu'))
        model.eval()

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        # Classes possibles
        labels = {
            0: "ECG Normal",
            1: "Fibrillation Auriculaire (AFib)",
            2: "Infarctus du Myocarde (MI)",
            3: "Bloc de Branche Gauche (LBBB)",
            4: "Tachycardie Sinusale",
        }

        return labels.get(prediction, "Indéterminé")

    # Exécuter l'IA et afficher le diagnostic
    st.subheader("📊 Résultat de l'Interprétation AI")
    diagnosis = predict_ecg_condition(gray)
    st.success(f"🩺 **Diagnostic AI : {diagnosis}**")