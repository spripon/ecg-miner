import streamlit as st
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

st.title("ECG Miner - Web Version")

uploaded_file = st.file_uploader("Upload ECG file (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load ECG data
    data = pd.read_csv(uploaded_file)
    ecg_signal = data.iloc[:,0]

    st.subheader("Raw ECG Data")
    st.line_chart(ecg_signal)

    # Process ECG Signal (from original ECG Miner)
    sampling_rate = 250  # or adjust based on your data
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

    st.subheader("Analyzed ECG Signal")
    fig = nk.ecg_plot(signals, info)
    st.pyplot(fig)

    # Show ECG features summary
    st.subheader("ECG Metrics")
    ecg_summary = nk.ecg_intervalrelated(signals)
    st.write(ecg_summary)
