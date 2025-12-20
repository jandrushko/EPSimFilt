# MEP Filter Tool Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MEP Filter Testing Tool                   │
│                  (Streamlit Web Interface)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Modules (src/)                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Signal Generator │  │ Noise Generator  │                │
│  │                  │  │                  │                │
│  │ - MEP waveforms  │  │ - White noise    │                │
│  │ - Double peaks   │  │ - Line noise     │                │
│  │ - Baseline EMG   │  │ - EMG noise      │                │
│  │ - Templates      │  │ - ECG artifacts  │                │
│  └──────────────────┘  │ - Movement       │                │
│                         │ - TMS artifacts  │                │
│                         └──────────────────┘                │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │    Filters       │  │     Metrics      │                │
│  │                  │  │                  │                │
│  │ - Butterworth    │  │ - Amplitude      │                │
│  │ - FIR (various)  │  │ - Timing         │                │
│  │ - Notch          │  │ - Morphology     │                │
│  │ - Moving average │  │ - Correlation    │                │
│  │ - Savitzky-Golay │  │ - SNR            │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dependencies Layer                        │
├─────────────────────────────────────────────────────────────┤
│  NumPy │ SciPy │ Matplotlib │ Pandas │ Seaborn │ Streamlit │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Signal Generation Pipeline
```
User Parameters → MEPGenerator → Clean MEP Signal
                                       │
                                       ▼
               NoiseGenerator → Noisy MEP Signal
                                       │
                                       ▼
                    Store in session_state
```

### 2. Filter Testing Pipeline
```
Noisy Signal + Filter Config → MEPFilters → Filtered Signal
                                                    │
                                                    ▼
        Clean + Filtered + Time → MEPMetrics → Performance Metrics
                                                    │
                                                    ▼
                              Visualization + Display
```

### 3. Batch Analysis Pipeline
```
Parameters → Generate Test Matrix
                   │
                   ▼
         For each configuration:
                   │
                   ├─> Generate signal
                   ├─> Add noise (multiple times)
                   ├─> Apply filter
                   ├─> Calculate metrics
                   │
                   ▼
         Aggregate Results
                   │
                   ▼
    Statistical Summary + Visualizations + Export
```

## Module Responsibilities

### signal_generator.py
**Purpose**: Generate realistic MEP waveforms
**Key Functions**:
- `generate_mep()` - Standard MEP with configurable parameters
- `generate_double_peak_mep()` - Dual-peak MEPs
- `generate_template_based_mep()` - Use real MEP as template
- `add_baseline_emg()` - Pre-stimulus muscle activity

**Output**: Time vector + clean MEP signal (numpy arrays)

### noise_generator.py
**Purpose**: Add realistic noise and artifacts
**Key Functions**:
- `add_white_noise()` - Gaussian noise with SNR control
- `add_line_noise()` - 50/60 Hz with harmonics
- `add_emg_noise()` - Filtered noise (20-500 Hz)
- `add_ecg_artifact()` - QRS complexes
- `add_movement_artifact()` - Low-frequency drift
- `add_tms_artifact()` - Stimulus-related spike
- `add_composite_noise()` - Combine multiple noise types

**Output**: Noisy MEP signal (numpy array)

### filters.py
**Purpose**: Apply digital filters
**Key Functions**:
- `butterworth_filter()` - Zero-phase or single-pass
- `fir_filter()` - Various window types
- `notch_filter()` - Line noise removal
- `moving_average_filter()` - Simple smoothing
- `savitzky_golay_filter()` - Polynomial smoothing
- `apply_filter_cascade()` - Apply multiple filters
- `get_frequency_response()` - Calculate filter response

**Output**: Filtered MEP signal (numpy array)

### metrics.py
**Purpose**: Evaluate filter performance
**Key Functions**:
- `detect_mep_onset()` - Threshold-based detection
- `calculate_peak_amplitude()` - Peak-to-peak measurement
- `calculate_peak_latency()` - Time of peak
- `calculate_area_under_curve()` - Integrate MEP
- `calculate_snr()` - Signal-to-noise ratio
- `calculate_correlation()` - Compare with ground truth
- `calculate_rmse()` - Root mean square error
- `calculate_all_metrics()` - Comprehensive evaluation
- `detect_spurious_oscillations()` - Ringing detection

**Output**: Dictionary of metrics

### app.py
**Purpose**: GUI interface and user interaction
**Structure**:
- Tab 1: Signal Generation
- Tab 2: Filter Testing
- Tab 3: Batch Analysis
- Tab 4: About/Documentation

**Features**:
- Interactive parameter sliders
- Real-time visualization
- Session state management
- Results export

## State Management

```
session_state:
  ├─ generated_signals:
  │    ├─ time (array)
  │    ├─ mep_clean (array)
  │    ├─ mep_noisy (array)
  │    ├─ sampling_rate (int)
  │    └─ parameters (dict)
  │
  ├─ filter_results: (list)
  │    └─ [result1, result2, ...]
  │         ├─ filter_params (dict)
  │         ├─ filtered_signal (array)
  │         └─ metrics (dict)
  │
  └─ batch_results: (DataFrame)
       └─ columns: config_label, metrics, iterations...
```

## Key Design Decisions

### 1. Modular Architecture
- **Why**: Each module can be used independently or through GUI
- **Benefit**: Easy to test, maintain, and extend
- **Trade-off**: Slightly more complex than monolithic design

### 2. Zero-Phase Filtering Default
- **Why**: Preserves MEP timing - critical for latency analysis
- **Implementation**: Uses `scipy.signal.filtfilt()`
- **Trade-off**: Not suitable for real-time applications

### 3. Session State for Signal Storage
- **Why**: Avoid regenerating signals on every interaction
- **Benefit**: Faster, more responsive interface
- **Trade-off**: Memory usage for long sessions

### 4. Streamlit for GUI
- **Why**: Rapid development, modern interface, web-based
- **Benefit**: No installation complexity, cross-platform
- **Trade-off**: Requires internet browser

### 5. Comprehensive Metrics
- **Why**: Different applications prioritize different aspects
- **Benefit**: Users can optimize for their specific needs
- **Trade-off**: More complexity in interpretation

## Extension Points

### Easy to Add:
1. **New Filter Types**
   - Add method to `filters.py`
   - Update `apply_filter_cascade()`
   - Add to GUI dropdown

2. **New Noise Types**
   - Add method to `noise_generator.py`
   - Add checkbox in GUI
   - Include in `add_composite_noise()`

3. **New Metrics**
   - Add method to `metrics.py`
   - Update `calculate_all_metrics()`
   - Display in results

4. **Custom MEP Templates**
   - Load from file in Signal Generation tab
   - Use `generate_template_based_mep()`

### Moderate Complexity:
1. **Real-time Analysis Mode**
   - Implement single-pass filters
   - Add streaming data support
   - Adjust metrics for causal processing

2. **Machine Learning Integration**
   - Add automated filter recommendation
   - Train on optimal configurations
   - Implement in separate module

3. **Database Integration**
   - Store results in SQL database
   - Add historical comparison
   - Track filter evolution

### Advanced:
1. **Multi-channel Analysis**
   - Extend for HD-EMG arrays
   - Add spatial filtering options
   - Implement cross-channel metrics

2. **Adaptive Filtering**
   - Implement Wiener filters
   - Add Kalman filtering
   - Dynamic parameter adjustment

3. **Cloud Deployment**
   - Deploy on Streamlit Cloud
   - Add user authentication
   - Implement result sharing

## Performance Considerations

### Memory Usage
- **Clean signal**: ~1 KB (200 ms at 2 kHz)
- **Session state**: ~10-100 KB depending on history
- **Batch results**: ~1 MB for 1000 tests
- **Recommendation**: Clear results periodically for long sessions

### Computation Time
- **Signal generation**: <10 ms
- **Noise addition**: ~20 ms
- **Single filter application**: ~50 ms
- **Metrics calculation**: ~100 ms
- **Batch analysis (100 tests)**: ~10-20 seconds

### Optimization Opportunities
1. Vectorize batch analysis (parallel processing)
2. Cache frequency responses
3. Use pre-computed filter coefficients
4. Implement progressive loading for batch results

## Security Considerations

### Safe by Design
- No file system access beyond designated directories
- No external API calls
- No code execution from user input
- Session state isolated per user

### Recommendations for Deployment
1. Use HTTPS for web deployment
2. Implement rate limiting for batch analysis
3. Add user authentication if sharing
4. Regular dependency updates

## Testing Strategy

### Unit Tests (test_modules.py)
- Import verification
- Signal generation
- Noise addition
- Filtering operations
- Metrics calculation

### Integration Tests
- Full pipeline execution
- GUI interaction simulation
- Results validation

### Validation Tests
- Compare with literature values
- Test on known signals
- Verify against established methods

---

This architecture enables:
✅ Easy maintenance and updates
✅ Independent module testing
✅ Flexible extension
✅ Clear separation of concerns
✅ Production-ready deployment
