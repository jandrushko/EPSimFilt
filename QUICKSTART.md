# Quick Start Guide

## Installation (5 minutes)

### Option 1: Quick Install
```bash
pip install streamlit numpy scipy matplotlib seaborn pandas
streamlit run app.py
```

### Option 2: Virtual Environment (Recommended)
```bash
# Create and activate virtual environment
python -m venv mep_env
source mep_env/bin/activate  # On Mac/Linux
# OR
mep_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## First Use (10 minutes)

### 1. Generate Your First Signal (2 min)
1. Open the **Signal Generation** tab
2. Use default parameters (or adjust to your needs)
3. Click **"Generate Signal"** button
4. Review the clean and noisy signals

### 2. Test a Filter (3 min)
1. Switch to **Filter Testing** tab
2. Select filter type: **butterworth**
3. Set high-pass: **10 Hz**
4. Set low-pass: **500 Hz**
5. Filter order: **4**
6. Click **"Apply Filter"** button
7. Review the results and metrics

### 3. Compare Multiple Filters (5 min)
1. Try different filter types:
   - Butterworth (orders 2, 4, 6)
   - FIR with Hamming window
   - FIR with Hann window
2. Compare their performance metrics
3. Note which performs best for your criteria

## Typical Workflow

### For Method Development
```
1. Generate signal with realistic parameters
   ↓
2. Add noise matching your recording conditions
   ↓
3. Test 3-5 candidate filters interactively
   ↓
4. Run batch analysis on top performers
   ↓
5. Export results and select optimal filter
```

### For Validation Study
```
1. Generate signals across range of amplitudes (0.5-2 mV)
   ↓
2. Test each with multiple SNR levels (5, 10, 15, 20 dB)
   ↓
3. Run batch analysis with all filter candidates
   ↓
4. Identify filter with best overall performance
   ↓
5. Export comprehensive results for manuscript
```

## Key Settings to Adjust

### For Realistic MEPs
- **Amplitude**: 0.5-2 mV (typical range)
- **Duration**: 25-35 ms (flexors) or 20-30 ms (extensors)
- **Onset Latency**: 18-22 ms (upper limb)
- **SNR**: 5-15 dB (typical for single trials)

### For Conservative Filtering
- **High-pass**: 10-20 Hz (avoid DC drift)
- **Low-pass**: 500-1000 Hz (preserve MEP content)
- **Order**: 4th (standard) or 2nd (gentler)
- **Notch**: Only if line noise is severe

### For Aggressive Noise Reduction
- **High-pass**: 20 Hz
- **Low-pass**: 250 Hz
- **Order**: 6th-8th
- **Notch**: Enabled with narrow bandwidth

## Common Issues & Solutions

**Issue**: Filter causes timing shifts
- **Solution**: Ensure using zero-phase filtering (default for Butterworth)
- **Check**: Look at "peak_latency_error_ms" metric

**Issue**: Filter creates oscillations
- **Solution**: Reduce filter order or use FIR instead
- **Check**: Look at baseline in filtered signal

**Issue**: Not enough noise reduction
- **Solution**: Increase filter order or use narrower bandwidth
- **Trade-off**: May increase distortion

**Issue**: Amplitude underestimated after filtering
- **Solution**: Use gentler filter (lower order, wider bandwidth)
- **Check**: "amplitude_error_pct" metric

## Tips for Best Results

1. **Match simulation to reality**: Set SNR and noise types to match your recordings
2. **Test multiple criteria**: Don't optimize for amplitude only - consider timing too
3. **Use batch analysis**: Single tests can be misleading due to noise variability
4. **Document everything**: Export results and save filter configurations
5. **Validate on real data**: After identifying candidate filters, test on actual MEPs

## Next Steps

1. Read the full **README.md** for detailed documentation
2. Check the **About** tab in the app for technical details
3. Run **example_usage.py** to see programmatic usage
4. Adapt filters to your specific research questions

## Getting Help

- Check **README.md** for comprehensive guide
- Review **About** tab for technical background
- Run example scripts to understand modules
- Contact: justin.andrushko@northumbria.ac.uk

---

**Ready to start? Run: `streamlit run app.py`**
