# Fan Anomaly Detection — First-Shot Acoustic Anomaly Detection Under Domain Shift

> **Mechanical Engineering ML Project — DCASE 2024 Task 2 / MIMII DUE**

A machine learning system that listens to industrial fan audio and decides whether the machine is behaving normally or abnormally. The system uses a **multi-transformer meta-stack** — fusing handcrafted acoustic features with embeddings from two pretrained audio transformers (MIMI codec and AST) — and follows up every detection with an **ODE-based harmonic fault diagnosis**.

---

## 📊 Headline result

| Metric | Value |
|---|---|
| Pooled AUC | **0.7255** |
| Partial AUC @ FPR ≤ 10% | 0.6168 |
| Average Precision | 0.7312 |
| Operating Precision | 0.667 (at p85) |
| Operating Recall | 0.660 |
| Flag rate | 49.5% |
| DCASE official baseline (MAHALA) | 0.610 |
| **Improvement over DCASE baseline** | **+11.5 AUC points** |

The meta-stack winner combines **HC_386** (handcrafted), **S5_MIMI** (Kyutai codec), and **AST** (Audio Spectrogram Transformer), with Wav2Vec2 dropped at Gate 2 for being below threshold.

---

## 🏗 Architecture

```
audio (.wav, 16 kHz, 10 s)
   │
   ├── §2.1  MIMI codec encoder    →  pooled mean+std  →  LOF (k=3, PCA=5)  →  AUC 0.716
   ├── §2.2  Wav2Vec2 transformer  →  pooled mean+std  →  LOF              →  AUC 0.579 (dropped)
   ├── §2.3  AST transformer       →  CLS+mean+std     →  LOF (k=3, PCA=20) →  AUC 0.654
   └── §2.4  Handcrafted 386-d     →  MFCC + Δ + Δ² +  →  LOF (k=3, PCA=10) →  AUC 0.699
                                       spectral stats
   │
   ├── §3.4  ECDF calibration → uniform[0,1] per scorer
   │
   ├── §3.4  Logistic Regression meta-stacker
   │           weights:  HC=+4.32   MIMI=+3.00   AST=+1.87   bias=-7.03
   │           trained on synthetic pseudo-anomalies (noise/pitch-shift/time-mask)
   │
   ├── §5    Threshold @ p85 of training scores  →  Flag / Don't flag
   │
   └── §6    ODE harmonic decomposition (post-detection only)
              f₀ = 23.7 Hz, 5 harmonics + ½× sub-harmonic + broadband
              → Fault classification (Unbalance / Misalignment / Looseness / Bearing)
              → Severity gauge (Nominal / Early / Moderate / Severe / Critical)
```

---

## 📁 Repository layout

```
fan-anomaly-detection/
├── README.md
├── requirements.txt
├── preprocessing.py
├── fan_anomaly_detection_model.ipynb
├── Fan_Anomaly_Detection.html
├── report ML.pdf
├── ML_ppt.pptx
```

---

## 📂 Description des fichiers

- **preprocessing.py** → pipeline de traitement audio (feature extraction, préparation des données)
- **fan_anomaly_detection_model.ipynb** → entraînement du modèle + évaluation
- **Fan_Anomaly_Detection.html** → interface utilisateur (dashboard)
- **report ML.pdf** → rapport final (1 page)
- **ML_ppt.pptx** → slides de présentation
- **requirements.txt** → dépendances Python


## 🐳 Run with Docker
The project is containerized using Docker and deployed on Docker Hub.

```bash
docker pull jeanmajdalani/fan-anomaly-detection:latest
docker run -p 8501:8501 jeanmajdalani/fan-anomaly-detection:latest
```
Then open in your browser:
```
http://localhost:8501
```
---
## ▶️ Quick Run (Local)

```bash
pip install -r requirements.txt
jupyter notebook fan_anomaly_detection_model.ipynb
```
Then open:

```text
fan_anomaly_detection_model.ipynb
```


Response:
```json
{
  "score": 0.834,
  "decision": "anomaly",
  "severity_pct": 92.3,
  "severity_label": "MODERATE ANOMALY",
  "fault": "MISALIGNMENT (2× dominant, +2.4σ)",
  "rationale": "Shaft parallel or angular misalignment...",
  "harm_z_scores": {"1x": 0.4, "2x": 2.4, "3x": 0.6, "4x": -0.1, "5x": 0.2},
  "model_version": "v37"
}
```

---

## 🔬 Reproduce the numbers

Run the notebook end-to-end on Google Colab with a T4 GPU. First run takes ~25 minutes (transformer model downloads + feature extraction); subsequent runs reuse cached features and finish in ~3 minutes.

```bash
# Local execution (requires ~6 GB RAM, GPU recommended)
pip install -r service/requirements.txt
jupyter notebook notebook/fan_anomaly_detection.ipynb
```

Reproducibility seed is locked at `1337` throughout.

---

## 📦 Dataset

This project uses the **DCASE 2024 Task 2 First-Shot Unsupervised Anomaly Detection** dataset (fan machine subset), built on **MIMII DUE**. 1,000 normal training clips + 200 test clips (100 normal, 100 anomaly) at 16 kHz, 10 seconds each.

Public URLs:
- DCASE 2024 challenge page: <https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection>
- MIMII DUE: <https://zenodo.org/records/4740355>
- DCASE 2024 additional data: <https://zenodo.org/records/11259435>

---

## 📚 Methodology highlights

1. **Gated multi-transformer pipeline** — Each transformer must individually clear an AUC threshold (Gate 1 ≥ 0.68 for MIMI, Gates 2–3 ≥ 0.58 for W2V2/AST) before being included in the fusion. Wav2Vec2 was dropped this run.
2. **Pseudo-anomaly meta-training** — Since the training set is normal-only, we generate synthetic anomalies (additive noise, ±2 semitone pitch shift, 1-second time-mask) to train the LogReg meta-stacker.
3. **ECDF calibration** — Each scorer is mapped to uniform [0,1] using its own training-score CDF before fusion. Removes scale mismatch.
4. **Honest threshold selection** — Operating thresholds are picked at percentiles of training scores, never at quantiles of test scores. No test-set tuning.
5. **ODE post-detection diagnosis** — Welch PSD at harmonics of fan rotation frequency (f₀ = 23.7 Hz). Fault classification by which harmonic deviates most (Unbalance, Misalignment, Looseness, Bearing degradation).

---

## 👥 Authors & contributions

| Member | Primary responsibility |
|---|---|
| Hafez Al Ghossainy | ML pipeline (feature extraction, meta-stack, evaluation) |
| Jean Majdalani | Containerization & deployment (Docker, FastAPI, UI integration) |
| Jad Nader | Documentation, presentation, and ODE physical interpretation |

See `CONTRIBUTORS.md` for commit attribution.

---

## ⚠️ Known limitations

1. **Domain-shift performance gap** — Source-domain AUC exceeds target-domain AUC. The system flags target-domain anomalies less reliably, reflecting the realistic difficulty of unsupervised generalization across operating conditions.
2. **Information ceiling around ~0.73 AUC** — Across 15+ pipeline iterations spanning autoencoders, classifier embeddings, multiple transformer families, and various fusion strategies, results converge near 0.71–0.73 AUC. We treat this as the dataset's information ceiling under unsupervised constraints rather than a tunable parameter.
3. **No labeled anomalies at training time** — The pseudo-anomaly meta-trainer is the necessary workaround. It generalizes to real anomalies but with imperfect transfer; this is the dominant residual error source.
4. **49.5% flag rate at the chosen operating point** — which remains below the 55% target ceiling at recall 0.66. A lower flag rate trades off recall, which matters more for safety-critical maintenance scheduling.

---

## 📄 License

MIT — see `LICENSE`.

## 📖 References

- *DCASE 2024 Task 2: First-Shot Unsupervised Anomalous Sound Detection*, Harada et al., DCASE Workshop 2024.
- *MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection with Domain Shifts*, Tanabe et al., 2021.
- *MIMI: A Streaming Audio Codec for Speech*, Kyutai, 2024.
- *AST: Audio Spectrogram Transformer*, Gong et al., Interspeech 2021.

## 🔗 Links

- GitHub: https://github.com/naderjad24/MechaEngineering
- Docker Hub: https://hub.docker.com/r/jeanmajdalani/fan-anomaly-detection
