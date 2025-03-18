st.subheader("📌 Sélectionnez le format et la disposition des dérivations ECG")

ecg_format = st.selectbox(
    "Format de l'ECG",
    ["1 page, 12x1", "2 pages, 6x1", "1 page, 6x2", "1 page, 6x2 avec dérivation rythmique", "1 page, 3x4"]
)

st.write(f"🖼️ Format sélectionné : **{ecg_format}**")