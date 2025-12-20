# MEPSimFilt

## 🧠 A Systematic Digital Filter Evaluation tool for Motor Evoked Potentials

**Version:** 1.0.0  
**Author:** Justin W. Andrushko, PhD  
**Institution:** Northumbria University  
**Initiative:** TMSMultiLab

---

## Overview

The MEPSimFilt is a specialised Python application for systematic evaluation of digital filter parameters in transcranial magnetic stimulation (TMS) research. It addresses the critical methodological gap created by substantial heterogeneity in filtering approaches across published studies (filter parameters ranging from 4-1500 Hz bandpass configurations).

This tool enables researchers to:
- Generate realistic Motor Evoked Potential (MEP) waveforms with multiple morphologies
- Simulate comprehensive physiological and technical noise
- Systematically evaluate filter configurations across thousands of iterations
- Perform rigorous statistical analysis with complete assumption testing
- Generate publication-ready figures and formal statistical statements

---

## Key Features

### Signal Generation
- **5 MEP morphologies:** Monophasic, bi-phasic, tri-phasic, double peak, with baseline EMG
- **Realistic temporal dynamics:** Smooth 3 ms sigmoid onset ramps, configurable rise/decay times
- **Pre-stimulus windows:** -20 ms baseline for artefact contextualization
- **Flexible parameters:** Amplitude (0.1-5.0 mV), duration (20-200 ms), latency (10-50 ms)
- **High sampling rates:** Up to 10,000 Hz

### Noise Modelling
- **6 noise types:**
  - White Gaussian noise
  - Electromyographic noise (20-500 Hz, burst events)
  - Line noise (50/60 Hz + harmonics)
  - Electrocardiographic artefacts
  - Movement artefacts (<2 Hz drift)
  - TMS artefact (1.5 mV residual post-blanking)
- **SNR control:** -10 to +40 dB
- **Composite or individual:** Enable any combination

### Filter Evaluation
- **Butterworth IIR filters:** 1st through 8th orders (effective 2nd-16th with zero-phase filtfilt)
- **FIR filters:** Hamming, Hann, Blackman windows
- **Notch filters:** 50/60 Hz line noise rejection
- **Comprehensive testing:** Up to 10,000 iterations per configuration
- **Batch processing:** Test up to 96 configurations automatically

### Statistical Analysis
- **Assumption testing:**
  - Shapiro-Wilk normality tests (all groups)
  - Levene's test for variance homogeneity
  - Sample size adequacy assessment
- **Automatic test selection:**
  - One-way ANOVA (parametric) when assumptions met
  - Welch's ANOVA (unequal variances)
  - Kruskal-Wallis H-test (non-parametric) when violated
- **Complete rationale:** Full transparency in statistical decisions
- **Formal reporting:** APA-style publication-ready statements
- **Effect sizes:** Cohen's d for all comparisons
- **Multiple comparison correction:** Bonferroni adjustment

### Visualization
- **Enhanced box plots:** Color and pattern coding by filter type/order
- **Multi-filter overlays:** Compare up to 7 configurations with z-ordering
- **Heatmaps:** Performance color-coded, sorted by error magnitude
- **Time-frequency analysis:** Morlet wavelet spectrograms (improved implementation)
- **High-quality export:** 150/300/600 DPI options for all plots

### Documentation
- **Adaptive methods text:** Three detail levels (Standard/Detailed/Complete)
- **Parameter documentation:** All signal, noise, and filter parameters
- **Statistical methods:** Complete assumption testing and analysis description
- **RIS export:** EndNote/Zotero compatible references
- **Downloadable reports:** Assumption testing, formal statements, full analysis

---

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# 1. Extract the tool
unzip MEP_Filter_Tool_v1.0.0.zip
cd mep_filter_tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

### Dependencies
- streamlit ≥1.28.0
- numpy ≥1.24.0
- scipy ≥1.11.0
- matplotlib ≥3.7.0
- pandas ≥2.0.0
- seaborn ≥0.12.0

---

## Quick Start

### Basic Workflow

1. **Signal Generation Tab:**
   - Select MEP morphology
   - Configure amplitude, duration, noise
   - Generate signal

2. **Filter Testing Tab:**
   - Choose filter type and parameters
   - Apply and review metrics
   - Iterate to explore parameter space

3. **Batch Analysis Tab:**
   - Define filter configurations to test
   - Set iteration count (≥1024 for publication)
   - Run batch analysis
   - Export CSV results

4. **Statistical Analysis:**
   - Review assumption testing
   - Examine test selection rationale
   - Download formal statements
   - Generate comparison tables

5. **Visualization:**
   - Create publication-quality figures
   - Export at appropriate DPI (600 for print)

6. **Documentation:**
   - Generate methods text
   - Download statistical reports
   - Export references

---

## Statistical Methodology

### Assumption Testing

Prior to hypothesis testing, three critical assumptions are evaluated:

1. **Normality:** Shapiro-Wilk tests for each configuration group
2. **Variance Homogeneity:** Levene's test across all groups
3. **Sample Size:** Adequacy assessment (n ≥ 30 recommended)

### Test Selection

The appropriate statistical test is automatically selected based on assumption test results:

| Assumptions Status | Selected Test | Post-hoc |
|-------------------|---------------|----------|
| All met (normal + homogeneous + adequate n) | One-way ANOVA | Tukey HSD |
| Normal + heterogeneous variances | Welch's ANOVA | Games-Howell |
| Non-normal or small n | Kruskal-Wallis H-test | Mann-Whitney U + Bonferroni |

Complete rationale provided for every statistical decision.

### Formal Reporting

APA-style statements automatically generated:

```
"Assumption testing was conducted to determine appropriate statistical 
methods. Shapiro-Wilk tests indicated [RESULTS]. Levene's test [RESULTS]. 
Based on these findings, [SELECTED TEST] was employed.

A [TEST NAME] revealed significant differences in [METRIC] between 
configurations ([STATISTIC] = [VALUE], p < .001). The optimal configuration 
was [CONFIG], demonstrating mean [METRIC] of [MEAN] ± [SD], median = [MEDIAN], 
95% CI [LOWER, UPPER]."
```

Copy directly into manuscript Methods and Results sections.

---

## File Structure

```
mep_filter_tool/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── assets/
│   └── TMSMultiLab_logo.png   # TMSMultiLab branding
└── src/
    ├── signal_generator.py     # MEP signal synthesis
    ├── noise_generator.py      # Noise modelling
    ├── filters.py              # Digital filter implementations
    └── metrics.py              # Performance metrics
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{andrushko2025mep,
  author = {Andrushko, Justin W.},
  title = {MEP Filter Testing Tool: Systematic Digital Filter Evaluation 
           for Motor Evoked Potentials},
  year = {2025},
  version = {1.0.0},
  publisher = {Northumbria University},
  url = {https://github.com/andrushko/mep-filter-tool}
}
```

---

## TMSMultiLab

This tool is part of the TMSMultiLab initiative for advancing methodological standards in transcranial magnetic stimulation research.

**Learn more:** https://github.com/TMSMultiLab/TMSMultiLab/wiki

---

## License

[To be specified]

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

---

## Support

**Email:** justin.andrushko@northumbria.ac.uk  
**GitHub:** https://github.com/andrushko/mep-filter-tool  
**TMSMultiLab:** https://github.com/TMSMultiLab

---

## Acknowledgments

Developed at Northumbria University as part of the TMSMultiLab collaborative initiative for improving TMS methodological standards and reproducibility.

---

**Version 1.0.0 | December 2025 | TMSMultiLab Initiative**
