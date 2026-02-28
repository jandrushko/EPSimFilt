# EPSimFilt 🧠

**Systematic Digital Filter Evaluation for Evoked Potentials**

Version 1.0.0 | Justin W. Andrushko, PhD | Northumbria University

---

## Quick Deploy to Streamlit Cloud (Free, ~5 minutes)

### 1. Push to GitHub

```
EPSimFilt/
├── app.py
├── signal_generator.py
├── noise_generator.py
├── filters.py
├── metrics.py
├── methods_template.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── assets/
    └── TMSMultiLab_logo.png
```

```bash
git init
git add .
git commit -m "Initial EPSimFilt deployment"
git remote add origin https://github.com/YOUR_USERNAME/EPSimFilt.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Deploy!"**

Your app will be live at `https://YOUR_USERNAME-epsimfilt-app-XXXX.streamlit.app`

---

## Run Locally

```bash
# Clone or download the repo
git clone https://github.com/YOUR_USERNAME/EPSimFilt.git
cd EPSimFilt

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

App opens automatically at `http://localhost:8501`

---

## Features

- **Signal Generation** — Mono/bi/tri-phasic EPs with configurable parameters
- **Real Waveform Loading** — LabChart, Spike2, LabView, CSV, MAT formats
- **Noise Models** — EMG, line noise, ECG, TMS artefact, movement
- **Filter Testing** — Butterworth (1–8th order), FIR, notch filters
- **Batch Analysis** — Up to 10,000 iterations per configuration
- **Time-Frequency** — Morlet wavelet analysis
- **Methods Generator** — Publication-ready methods text (3 detail levels)
- **Export** — CSV results, 300/600 DPI figures, RIS references

---

## Citation

Andrushko, J.W. (2025). *EPSimFilt: A Systematic Digital Filter Evaluation tool for Evoked Potentials* (Version 1.0.0). Northumbria University.

Part of [TMSMultiLab](https://github.com/TMSMultiLab/TMSMultiLab/wiki)
