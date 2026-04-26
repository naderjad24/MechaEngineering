import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import tempfile
import os

SR = 16000
CLIP_SEC = 10

st.set_page_config(
    page_title="Fan Anomaly Detection",
    page_icon="🔊",
    layout="wide"
)

st.title("🔊 Fan Anomaly Detection Dashboard")
st.write("Industrial fan acoustic anomaly detection using audio signal analysis.")

st.markdown("---")

uploaded_file = st.file_uploader("Upload a fan audio file (.wav)", type=["wav"])

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SR, duration=CLIP_SEC, mono=True)
    target_len = SR * CLIP_SEC
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len], SR

def extract_features(y, sr):
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    return {
        "RMS Energy": rms,
        "Zero Crossing Rate": zcr,
        "Spectral Centroid": centroid,
        "Spectral Bandwidth": bandwidth,
        "Spectral Rolloff": rolloff,
        "Spectral Flatness": flatness
    }

def simple_anomaly_score(features):
    score = 0

    score += min(features["RMS Energy"] * 20, 1.0)
    score += min(features["Zero Crossing Rate"] * 8, 1.0)
    score += min(features["Spectral Flatness"] * 4, 1.0)
    score += min(features["Spectral Centroid"] / 4000, 1.0)

    score = score / 4
    return float(score)

def severity_label(score):
    if score < 0.45:
        return "NORMAL", "🟢"
    elif score < 0.65:
        return "WARNING", "🟡"
    elif score < 0.80:
        return "SEVERE", "🟠"
    else:
        return "CRITICAL", "🔴"

def fault_diagnosis(y, sr):
    freqs, psd = signal.welch(y, sr, nperseg=2048)
    dominant_freq = freqs[np.argmax(psd)]

    if dominant_freq < 40:
        return "Possible rotor unbalance"
    elif dominant_freq < 80:
        return "Possible shaft misalignment"
    elif dominant_freq < 150:
        return "Possible mechanical looseness"
    else:
        return "Possible bearing wear or high-frequency noise"

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    y, sr = load_audio(tmp_path)
    features = extract_features(y, sr)
    score = simple_anomaly_score(features)
    label, icon = severity_label(score)
    diagnosis = fault_diagnosis(y, sr)

    os.remove(tmp_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        t = np.linspace(0, len(y) / sr, len(y))
        ax.plot(t, y)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    with col2:
        st.subheader("Result")
        st.metric("Anomaly Score", f"{score:.3f}")
        st.metric("Decision", f"{icon} {label}")
        st.write("### Mechanical Diagnosis")
        st.info(diagnosis)

    st.markdown("---")

    st.subheader("Extracted Audio Features")
    st.table(features)

    st.subheader("Spectrogram")
    fig2, ax2 = plt.subplots()
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)

else:
    st.info("Please upload a .wav file to start the anomaly detection demo.")