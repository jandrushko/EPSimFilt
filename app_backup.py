"""
MEP Filter Testing Tool - Streamlit GUI
Interactive tool for testing and comparing digital filters for MEP analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signal_generator import MEPGenerator
from noise_generator import NoiseGenerator
from filters import MEPFilters
from metrics import MEPMetrics

# Page configuration
st.set_page_config(
    page_title="MEP Filter Testing Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_signals' not in st.session_state:
    st.session_state.generated_signals = None
if 'filter_results' not in st.session_state:
    st.session_state.filter_results = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# Title and description
st.title("🧠 MEP Filter Testing Tool")
st.markdown("""
**Systematically evaluate digital filter performance for Motor Evoked Potentials (MEPs)**

This tool allows you to:
- Generate realistic MEP signals with configurable parameters
- Add various types of physiological and technical noise
- Apply different filter types and compare their performance
- Evaluate filters using comprehensive metrics
- Run batch tests across parameter ranges
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎛️ Signal Generation", "🔧 Filter Testing", "📊 Batch Analysis", "📋 Methods", "ℹ️ About"])

# ===========================
# TAB 1: SIGNAL GENERATION
# ===========================
with tab1:
    st.header("Signal Generation & Noise Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MEP Parameters")
        sampling_rate = st.number_input("Sampling Rate (Hz)", 
                                       min_value=500, max_value=10000, 
                                       value=5000, step=100)
        
        mep_amplitude = st.slider("MEP Amplitude (mV)", 
                                 min_value=0.1, max_value=5.0, 
                                 value=1.0, step=0.1)
        
        mep_duration = st.slider("MEP Duration (ms)", 
                                min_value=10, max_value=80, 
                                value=30, step=5)
        
        onset_latency = st.slider("Onset Latency (ms)", 
                                 min_value=10, max_value=50, 
                                 value=20, step=5)
        
        rise_time = st.slider("Rise Time (ms)", 
                             min_value=3, max_value=20, 
                             value=8, step=1)
        
        decay_time = st.slider("Decay Time (ms)", 
                              min_value=5, max_value=30, 
                              value=15, step=1)
        
        asymmetry = st.slider("Asymmetry Factor", 
                             min_value=1.0, max_value=2.0, 
                             value=1.2, step=0.1)
        
        mep_type = st.selectbox("MEP Type", 
                               ["Standard", "Bi-phasic", "Tri-phasic", 
                                "Double Peak", "With Baseline EMG"])
        
    with col2:
        st.subheader("Noise Parameters")
        
        snr_db = st.slider("Overall SNR (dB)", 
                          min_value=-10, max_value=40, 
                          value=10, step=2)
        
        st.write("**Select Noise Types:**")
        include_white = st.checkbox("White Noise", value=True)
        include_line = st.checkbox("50 Hz Line Noise", value=True)
        include_emg = st.checkbox("EMG Noise", value=True)
        include_ecg = st.checkbox("ECG Artifact", value=False)
        include_movement = st.checkbox("Movement Artifact", value=False)
        include_tms = st.checkbox("TMS Artifact", value=False)
        
        if include_line:
            line_freq = st.selectbox("Line Frequency", [50, 60], index=0)
        else:
            line_freq = 50
            
    # Generate signal button
    if st.button("🎯 Generate Signal", type="primary"):
        with st.spinner("Generating signals..."):
            # Initialize generators
            mep_gen = MEPGenerator(sampling_rate=sampling_rate)
            noise_gen = NoiseGenerator(sampling_rate=sampling_rate)
            
            # Generate clean MEP
            if mep_type == "Standard":
                time, mep_clean = mep_gen.generate_mep(
                    amplitude=mep_amplitude,
                    duration=mep_duration/1000,
                    onset_latency=onset_latency/1000,
                    rise_time=rise_time/1000,
                    decay_time=decay_time/1000,
                    asymmetry=asymmetry
                )
            elif mep_type == "Bi-phasic":
                time, mep_clean = mep_gen.generate_biphasic_mep(
                    amplitude=mep_amplitude,
                    onset_latency=onset_latency/1000,
                    phase1_duration=rise_time/1000 * 1.2,
                    phase2_duration=decay_time/1000 * 1.0,
                    phase2_amplitude_ratio=0.8  # Strong negative phase for clear bi-phasic
                )
            elif mep_type == "Tri-phasic":
                time, mep_clean = mep_gen.generate_triphasic_mep(
                    amplitude=mep_amplitude,
                    onset_latency=onset_latency/1000,
                    phase1_duration=rise_time/1000 * 1.0,
                    phase2_duration=decay_time/1000 * 1.2,
                    phase3_duration=rise_time/1000 * 1.3,
                    phase2_ratio=0.75,  # Clear negative phase
                    phase3_ratio=0.4    # Visible positive recovery
                )
            elif mep_type == "Double Peak":
                time, mep_clean = mep_gen.generate_double_peak_mep(
                    amplitude1=mep_amplitude,
                    amplitude2=mep_amplitude * 0.6,
                    onset_latency=onset_latency/1000
                )
            else:  # With Baseline EMG
                time, mep_clean = mep_gen.generate_mep(
                    amplitude=mep_amplitude,
                    duration=mep_duration/1000,
                    onset_latency=onset_latency/1000,
                    rise_time=rise_time/1000,
                    decay_time=decay_time/1000,
                    asymmetry=asymmetry
                )
                mep_clean = mep_gen.add_baseline_emg(mep_clean, time, 
                                                    emg_amplitude=0.05,
                                                    onset_latency=onset_latency/1000)
            
            # Add noise
            mep_noisy = noise_gen.add_composite_noise(
                mep_clean, time,
                snr_db=snr_db,
                include_line=include_line,
                include_emg=include_emg,
                include_ecg=include_ecg,
                include_movement=include_movement,
                include_tms=include_tms
            )
            
            # Store in session state
            st.session_state.generated_signals = {
                'time': time,
                'mep_clean': mep_clean,
                'mep_noisy': mep_noisy,
                'sampling_rate': sampling_rate,
                'parameters': {
                    'amplitude': mep_amplitude,
                    'duration': mep_duration,
                    'onset_latency': onset_latency,
                    'snr_db': snr_db
                }
            }
            
        st.success("✅ Signal generated successfully!")
    
    # Display generated signals
    if st.session_state.generated_signals is not None:
        st.subheader("Generated Signals")
        
        data = st.session_state.generated_signals
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Plot clean MEP
        axes[0].plot(data['time']*1000, data['mep_clean'], 'b-', linewidth=1.5, label='Clean MEP')
        axes[0].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1, label='TMS Stimulus')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].set_title('Clean MEP Signal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot noisy MEP
        axes[1].plot(data['time']*1000, data['mep_noisy'], 'r-', linewidth=0.8, alpha=0.7, label='Noisy MEP')
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1, label='TMS Stimulus')
        axes[1].set_ylabel('Amplitude (mV)')
        axes[1].set_title('Noisy MEP Signal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot overlay
        axes[2].plot(data['time']*1000, data['mep_clean'], 'b-', linewidth=1.5, alpha=0.7, label='Clean')
        axes[2].plot(data['time']*1000, data['mep_noisy'], 'r-', linewidth=0.8, alpha=0.5, label='Noisy')
        axes[2].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1, label='TMS Stimulus')
        axes[2].set_xlabel('Time (ms) - Negative = Pre-Stimulus')
        axes[2].set_ylabel('Amplitude (mV)')
        axes[2].set_title('Overlay Comparison')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add download button for signal plots
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="💾 Download Signal Plots (300 DPI)",
            data=buf,
            file_name="generated_signals_300dpi.png",
            mime="image/png",
            key='download_signal_plots'
        )
        
        plt.close()
        
        # Display parameters
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Amplitude", f"{data['parameters']['amplitude']:.2f} mV")
        col2.metric("Duration", f"{data['parameters']['duration']:.0f} ms")
        col3.metric("Onset", f"{data['parameters']['onset_latency']:.0f} ms")
        col4.metric("SNR", f"{data['parameters']['snr_db']:.0f} dB")

# ===========================
# TAB 2: FILTER TESTING
# ===========================
with tab2:
    st.header("Interactive Filter Testing")
    
    if st.session_state.generated_signals is None:
        st.warning("⚠️ Please generate a signal in the 'Signal Generation' tab first!")
    else:
        # Initialize filters object for use throughout this tab
        data = st.session_state.generated_signals
        filters = MEPFilters(sampling_rate=data['sampling_rate'])
        
        # Calculate Nyquist frequency for validation
        nyquist_freq = data['sampling_rate'] / 2.0
        max_lowpass = int(nyquist_freq * 0.95)  # 95% of Nyquist as safe maximum
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Filter Configuration")
            
            filter_type = st.selectbox(
                "Filter Type",
                ["butterworth", "butterworth_single_pass", "fir_hamming", 
                 "fir_hann", "fir_blackman", "moving_average", "savitzky_golay"]
            )
            
            st.write("**Frequency Cutoffs**")
            use_highpass = st.checkbox("High-pass filter", value=True)
            if use_highpass:
                lowcut = st.slider("High-pass cutoff (Hz)", 
                                  min_value=1, max_value=100, value=10, step=1)
            else:
                lowcut = None
                
            use_lowpass = st.checkbox("Low-pass filter", value=True)
            if use_lowpass:
                default_highcut = min(500, max_lowpass)
                highcut = st.slider("Low-pass cutoff (Hz)", 
                                   min_value=50, max_value=max_lowpass, 
                                   value=default_highcut, step=50)
            else:
                highcut = None
            
            if 'butterworth' in filter_type:
                order = st.slider("Filter Order", 
                                 min_value=2, max_value=8, value=4, step=2)
            elif 'fir' in filter_type:
                order = 4  # Default for FIR
            elif filter_type == 'moving_average':
                window_size = st.slider("Window Size", 
                                       min_value=3, max_value=51, value=5, step=2)
                order = 4
            elif filter_type == 'savitzky_golay':
                window_length = st.slider("Window Length", 
                                         min_value=5, max_value=51, value=11, step=2)
                polyorder = st.slider("Polynomial Order", 
                                     min_value=2, max_value=5, value=3, step=1)
                order = 4
            else:
                order = 4
            
            use_notch = st.checkbox("50 Hz Notch Filter", value=False)
            
            if st.button("🔧 Apply Filter", type="primary"):
                # Prepare filter parameters
                filter_params = {
                    'filter_type': filter_type,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'order': order,
                    'notch_enabled': use_notch,
                    'notch_freq': 50
                }
                
                if filter_type == 'moving_average':
                    filter_params['window_size'] = window_size
                elif filter_type == 'savitzky_golay':
                    filter_params['window_length'] = window_length
                    filter_params['polyorder'] = polyorder
                
                # Apply filter
                with st.spinner("Applying filter..."):
                    mep_filtered = filters.apply_filter_cascade(
                        data['mep_noisy'], 
                        filter_params
                    )
                    
                    # Calculate metrics
                    metrics_calc = MEPMetrics(sampling_rate=data['sampling_rate'])
                    metrics = metrics_calc.calculate_all_metrics(
                        data['mep_clean'],
                        mep_filtered,
                        data['time']
                    )
                    
                    # Store results
                    result = {
                        'filter_params': filter_params,
                        'filtered_signal': mep_filtered,
                        'metrics': metrics
                    }
                    st.session_state.filter_results.append(result)
                    
                st.success("✅ Filter applied successfully!")
        
        # Display results
        if len(st.session_state.filter_results) > 0:
            with col2:
                st.subheader("Filter Results")
                
                # Get latest result
                result = st.session_state.filter_results[-1]
                data = st.session_state.generated_signals
                
                # Plot comparison
                fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Waveform comparison
                axes[0].plot(data['time']*1000, data['mep_clean'], 'b-', 
                           linewidth=1.5, alpha=0.7, label='Clean (Ground Truth)')
                axes[0].plot(data['time']*1000, data['mep_noisy'], 'gray', 
                           linewidth=0.8, alpha=0.4, label='Noisy')
                axes[0].plot(data['time']*1000, result['filtered_signal'], 'r-', 
                           linewidth=1.5, alpha=0.8, label='Filtered')
                axes[0].axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
                axes[0].set_ylabel('Amplitude (mV)')
                axes[0].set_title('Signal Comparison')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Error plot
                error = result['filtered_signal'] - data['mep_clean']
                axes[1].plot(data['time']*1000, error, 'r-', linewidth=1)
                axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
                axes[1].axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
                axes[1].set_xlabel('Time (ms) - Negative = Pre-Stimulus')
                axes[1].set_ylabel('Error (mV)')
                axes[1].set_title('Reconstruction Error')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Display metrics
                st.subheader("Performance Metrics")
                
                m = result['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Amplitude Error", 
                           f"{m['amplitude_error_pct']:.1f}%",
                           delta=f"{m['amplitude_error_abs']:.3f} mV")
                col2.metric("Peak Latency Error", 
                           f"{m['peak_latency_error_ms']:.2f} ms")
                col3.metric("Correlation", 
                           f"{m['correlation']:.3f}")
                col4.metric("RMSE", 
                           f"{m['rmse_mep']:.3f} mV")
                
                # Detailed metrics table
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Amplitude Error (%)',
                        'Peak Latency Error (ms)',
                        'Onset Error (ms)',
                        'AUC Error (%)',
                        'Correlation',
                        'RMSE (MEP window)',
                        'Baseline Std (mV)',
                        'SNR (dB)'
                    ],
                    'Value': [
                        f"{m['amplitude_error_pct']:.2f}",
                        f"{m['peak_latency_error_ms']:.3f}",
                        f"{m['onset_error_ms']:.3f}",
                        f"{m['auc_error_pct']:.2f}",
                        f"{m['correlation']:.4f}",
                        f"{m['rmse_mep']:.4f}",
                        f"{m['baseline_std']:.4f}",
                        f"{m['snr_filtered']:.2f}"
                    ]
                })
                
                st.dataframe(metrics_df, width='stretch', hide_index=True)
                
                # Frequency response
                st.subheader("Filter Frequency Response")
                freqs, response = filters.get_frequency_response(result['filter_params'])
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(freqs, response, 'b-', linewidth=2)
                ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude (dB)')
                ax.set_title('Filter Frequency Response')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ===========================
# TAB 3: BATCH ANALYSIS
# ===========================
with tab3:
    st.header("Batch Filter Analysis")
    st.markdown("""
    Test multiple filter configurations automatically and compare their performance.
    """)
    
    if st.session_state.generated_signals is None:
        st.warning("⚠️ Please generate a signal in the 'Signal Generation' tab first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Filter Configurations to Test")
            
            test_filters = st.multiselect(
                "Select Filter Types",
                ["butterworth", "fir_hamming", "fir_hann", "fir_blackman"],
                default=["butterworth", "fir_hamming"]
            )
            
            test_orders = st.multiselect(
                "Filter Orders (Butterworth only)",
                [2, 4, 6, 8],
                default=[2, 4]
            )
            
            st.write("**Highpass Cutoffs (Hz)**")
            highpass_cutoffs = st.text_input("Comma-separated values", "10, 20")
            
            st.write("**Lowpass Cutoffs (Hz)**")
            lowpass_cutoffs = st.text_input("Comma-separated values (lowpass)", "450, 500, 1000")
            
            st.info(f"ℹ️ **Note:** With sampling rate of {st.session_state.generated_signals['sampling_rate']} Hz, "
                   f"cutoff frequencies must be below {st.session_state.generated_signals['sampling_rate']/2:.0f} Hz (Nyquist limit).")
            
            include_notch_batch = st.checkbox("Test with/without 50 Hz notch", value=False)
            
        with col2:
            st.subheader("Analysis Options")
            
            n_iterations = st.slider("Iterations per configuration", 
                                    min_value=1, max_value=10000, value=10, step=1)
            
            st.info("""
            **Note:** Each iteration uses the same signal parameters but different 
            noise realisations to test robustness.
            """)
            
        if st.button("🚀 Run Batch Analysis", type="primary"):
            # Parse cutoff frequencies
            try:
                hp_cutoffs = [float(x.strip()) for x in highpass_cutoffs.split(',')]
                lp_cutoffs = [float(x.strip()) for x in lowpass_cutoffs.split(',')]
            except:
                st.error("Invalid cutoff frequencies. Please use comma-separated numbers.")
                st.stop()
            
            # Initialize
            data = st.session_state.generated_signals
            nyquist = data['sampling_rate'] / 2.0
            
            # Validate cutoff frequencies
            if any(lp >= nyquist for lp in lp_cutoffs):
                st.error(f"❌ Low-pass cutoffs must be less than Nyquist frequency ({nyquist:.0f} Hz). "
                        f"Please use values below {nyquist:.0f} Hz.")
                st.stop()
            
            if any(hp >= nyquist for hp in hp_cutoffs):
                st.error(f"❌ High-pass cutoffs must be less than Nyquist frequency ({nyquist:.0f} Hz). "
                        f"Please use values below {nyquist:.0f} Hz.")
                st.stop()
            
            filters = MEPFilters(sampling_rate=data['sampling_rate'])
            metrics_calc = MEPMetrics(sampling_rate=data['sampling_rate'])
            mep_gen = MEPGenerator(sampling_rate=data['sampling_rate'])
            noise_gen = NoiseGenerator(sampling_rate=data['sampling_rate'])
            
            # Create all filter configurations
            configurations = []
            for filter_type in test_filters:
                for hp in hp_cutoffs:
                    for lp in lp_cutoffs:
                        if 'butterworth' in filter_type:
                            for order in test_orders:
                                for notch in ([True, False] if include_notch_batch else [False]):
                                    configurations.append({
                                        'filter_type': filter_type,
                                        'lowcut': hp,
                                        'highcut': lp,
                                        'order': order,
                                        'notch_enabled': notch,
                                        'notch_freq': 50
                                    })
                        else:
                            for notch in ([True, False] if include_notch_batch else [False]):
                                configurations.append({
                                    'filter_type': filter_type,
                                    'lowcut': hp,
                                    'highcut': lp,
                                    'order': 4,
                                    'notch_enabled': notch,
                                    'notch_freq': 50
                                })
            
            # Run batch analysis
            results_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_tests = len(configurations) * n_iterations
            current_test = 0
            skipped_configs = []
            
            for config in configurations:
                config_failed = False
                for iteration in range(n_iterations):
                    current_test += 1
                    progress_bar.progress(current_test / total_tests)
                    status_text.text(f"Testing configuration {current_test}/{total_tests}...")
                    
                    try:
                        # Generate noisy signal
                        mep_noisy = noise_gen.add_composite_noise(
                            data['mep_clean'], data['time'],
                            snr_db=data['parameters']['snr_db'],
                            include_line=True,
                            include_emg=True
                        )
                        
                        # Apply filter
                        mep_filtered = filters.apply_filter_cascade(mep_noisy, config)
                        
                        # Calculate metrics
                        metrics = metrics_calc.calculate_all_metrics(
                            data['mep_clean'],
                            mep_filtered,
                            data['time']
                        )
                        
                        # Store results
                        result = {**config, **metrics, 'iteration': iteration}
                        results_list.append(result)
                        
                    except ValueError as e:
                        # Skip this configuration if it fails
                        if not config_failed:
                            config_failed = True
                            config_name = f"{config['filter_type']}_{config['lowcut']}-{config['highcut']}Hz"
                            skipped_configs.append((config_name, str(e)))
                        # Skip remaining iterations for this config
                        break
            
            progress_bar.empty()
            status_text.empty()
            
            # Show skipped configurations if any
            if skipped_configs:
                st.warning(f"⚠️ Skipped {len(skipped_configs)} configuration(s) due to errors:")
                for config_name, error in skipped_configs:
                    st.write(f"  • **{config_name}**: {error.split('.')[0]}")
                st.info("💡 **Tip:** Increase signal duration or use only Butterworth filters to avoid these issues.")
            
            # Convert to DataFrame
            if results_list:
                results_df = pd.DataFrame(results_list)
                st.session_state.batch_results = results_df
                
                successful_tests = len(results_list)
                st.success(f"✅ Batch analysis complete! Successfully completed {successful_tests} tests "
                          f"({len(configurations) - len(skipped_configs)} configurations × {n_iterations} iterations)")
            else:
                st.error("❌ No configurations completed successfully. Try increasing signal duration or using different filter types.")
                st.stop()
        
        # Display batch results
        if st.session_state.batch_results is not None:
            st.subheader("Batch Analysis Results")
            
            df = st.session_state.batch_results
            
            # Create configuration label
            df['config_label'] = (df['filter_type'].str[:7] + '_' + 
                                 df['lowcut'].astype(str) + '-' + 
                                 df['highcut'].astype(str) + 'Hz_' +
                                 'O' + df['order'].astype(str))
            if include_notch_batch:
                df['config_label'] += df['notch_enabled'].apply(lambda x: '_N' if x else '')
            
            # Aggregate metrics
            summary = df.groupby('config_label').agg({
                'amplitude_error_pct': ['mean', 'std'],
                'peak_latency_error_ms': ['mean', 'std'],
                'correlation': ['mean', 'std'],
                'rmse_mep': ['mean', 'std']
            }).round(3)
            
            st.subheader("Summary Statistics")
            st.dataframe(summary, width='stretch')
            
            # Visualizations
            st.subheader("Performance Comparison")
            
            # Create tabs for different visualization types
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📦 Box Plots", "🎨 Multi-Filter Overlay", "🔥 Heatmap"])
            
            # TAB 1: Enhanced Box Plots
            with viz_tab1:
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    metric_choice_box = st.selectbox(
                        "Select metric to visualise",
                        ['amplitude_error_pct', 'peak_latency_error_ms', 
                         'correlation', 'rmse_mep', 'baseline_std'],
                        key='boxplot_metric'
                    )
                
                with col_right:
                    st.write("**Plot Options:**")
                    show_grid = st.checkbox("Show grid", value=False, key='box_grid')
                    show_reference = st.checkbox("Show reference line", value=True, key='box_ref')
                    
                    # Color/pattern options
                    color_option = st.radio("Distinguish by:", 
                                           ["Filter type", "Filter order", "Both"], 
                                           index=0, key='box_color_opt')
                    
                    # Figure save options
                    st.write("**Save Options:**")
                    save_dpi = st.selectbox("Export DPI", [150, 300, 600], index=1, key='box_dpi')
                
                # Add reference line value input
                if show_reference:
                    if metric_choice_box == 'amplitude_error_pct':
                        ref_value = st.slider("Reference value (%)", -20.0, 20.0, 0.0, 0.5, key='box_ref_val')
                    elif metric_choice_box == 'peak_latency_error_ms':
                        ref_value = st.slider("Reference value (ms)", -5.0, 5.0, 0.0, 0.1, key='box_ref_val')
                    elif metric_choice_box == 'correlation':
                        ref_value = st.slider("Reference value", 0.0, 1.0, 0.95, 0.01, key='box_ref_val')
                    else:
                        ref_value = st.slider("Reference value", 0.0, 1.0, 0.1, 0.01, key='box_ref_val')
                else:
                    ref_value = 0
                
                # Create enhanced box plot
                fig, ax = plt.subplots(figsize=(14, 7))
                
                # Extract filter type and order for styling
                df['filter_base'] = df['filter_type'].apply(lambda x: x.split('_')[0] if '_' in x else x[:7])
                df['filter_order_str'] = 'O' + df['order'].astype(str)
                
                # Determine coloring/pattern scheme
                if color_option == "Filter type":
                    unique_filters = sorted(df['filter_base'].unique())
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_filters)))
                    filter_colors = dict(zip(unique_filters, colors))
                    use_patterns = False
                    
                elif color_option == "Filter order":
                    unique_orders = sorted(df['filter_order_str'].unique())
                    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_orders)))
                    order_colors = dict(zip(unique_orders, colors))
                    use_patterns = False
                    
                else:  # Both
                    unique_filters = sorted(df['filter_base'].unique())
                    unique_orders = sorted(df['order'].unique())
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_filters)))
                    filter_colors = dict(zip(unique_filters, colors))
                    patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo']
                    order_patterns = dict(zip(unique_orders, patterns[:len(unique_orders)]))
                    use_patterns = True
                
                # Create box plot data
                positions = []
                box_data = []
                box_colors = []
                box_patterns = []
                labels = []
                
                for i, (config, group) in enumerate(df.groupby('config_label')):
                    positions.append(i)
                    box_data.append(group[metric_choice_box].values)
                    filter_base = group['filter_base'].iloc[0]
                    filter_order = group['order'].iloc[0]
                    filter_order_str = group['filter_order_str'].iloc[0]
                    
                    if color_option == "Filter type":
                        box_colors.append(filter_colors[filter_base])
                        box_patterns.append('')
                    elif color_option == "Filter order":
                        box_colors.append(order_colors[filter_order_str])
                        box_patterns.append('')
                    else:  # Both
                        box_colors.append(filter_colors[filter_base])
                        box_patterns.append(order_patterns[filter_order])
                    
                    labels.append(config)
                
                bp = ax.boxplot(box_data, positions=positions, labels=labels, patch_artist=True)
                
                # Style the boxes
                for patch, color, pattern in zip(bp['boxes'], box_colors, box_patterns):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    if pattern:
                        patch.set_hatch(pattern)
                
                # Add reference line
                if show_reference:
                    ax.axhline(ref_value, color='red', linestyle='--', linewidth=2.5, 
                              alpha=0.8, label=f'Target: {ref_value}', zorder=1000)
                
                # Create appropriate legend
                if color_option == "Filter type":
                    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=filter_colors[f], 
                                                    alpha=0.7, label=f.capitalize()) for f in unique_filters]
                    ax.legend(handles=legend_elements, loc='best', title='Filter Type', fontsize=9)
                elif color_option == "Filter order":
                    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=order_colors[o], 
                                                    alpha=0.7, label=o) for o in unique_orders]
                    ax.legend(handles=legend_elements, loc='best', title='Filter Order', fontsize=9)
                else:  # Both
                    # Two-part legend: colors for type, patterns for order
                    color_legend = [plt.Rectangle((0,0),1,1, facecolor=filter_colors[f], 
                                                  alpha=0.7, label=f.capitalize()) for f in unique_filters]
                    pattern_legend = [plt.Rectangle((0,0),1,1, facecolor='gray', hatch=order_patterns[o],
                                                   alpha=0.7, label=f'Order {o}') for o in unique_orders]
                    
                    first_legend = ax.legend(handles=color_legend, loc='upper left', 
                                           title='Filter Type', fontsize=8)
                    ax.add_artist(first_legend)
                    ax.legend(handles=pattern_legend, loc='upper right', 
                             title='Order', fontsize=8)
                
                ax.set_xlabel('Filter Configuration', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric_choice_box.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_title(f'{metric_choice_box.replace("_", " ").title()} Distribution Across Iterations', 
                            fontsize=14, fontweight='bold')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                
                # Properly control grid visibility
                ax.grid(show_grid, alpha=0.3, which='both')
                if not show_grid:
                    ax.grid(False)
                    
                plt.suptitle('')  # Remove default pandas title
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add download button
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=save_dpi, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label=f"💾 Download Box Plot ({save_dpi} DPI)",
                    data=buf,
                    file_name=f"boxplot_{metric_choice_box}_{save_dpi}dpi.png",
                    mime="image/png"
                )
                
                plt.close()
            
            # TAB 2: Multi-Filter Overlay
            with viz_tab2:
                st.write("**Overlay multiple filter configurations to compare performance**")
                st.info("💡 **Tip:** Use z-ordering to control visibility - later selections appear on top")
                
                # Select multiple configurations
                available_configs = df['config_label'].unique().tolist()
                selected_configs = st.multiselect(
                    "Select configurations to overlay (up to 5 recommended):",
                    options=available_configs,
                    default=available_configs[:min(3, len(available_configs))] if len(available_configs) >= 3 else available_configs,
                    max_selections=7
                )
                
                col_overlay1, col_overlay2 = st.columns(2)
                with col_overlay1:
                    show_individual = st.checkbox("Show individual iterations", value=False, key='overlay_indiv')
                    if show_individual:
                        alpha_iterations = st.slider("Iteration transparency", 0.05, 0.5, 0.15, 0.05, key='overlay_alpha')
                    else:
                        alpha_iterations = 0.2
                
                with col_overlay2:
                    show_mean_only = st.checkbox("Mean lines only (fastest)", value=False, key='overlay_mean')
                    show_sd_envelope = st.checkbox("Show ±SD envelope", value=True, key='overlay_sd')
                    overlay_dpi = st.selectbox("Export DPI", [150, 300, 600], index=1, key='overlay_dpi')
                
                if len(selected_configs) > 0 and st.button("🎨 Generate Multi-Filter Overlay", type="primary", key='multi_overlay_btn'):
                    with st.spinner(f"Generating overlay for {len(selected_configs)} configurations..."):
                        # Initialize generators
                        noise_gen = NoiseGenerator(sampling_rate=data['sampling_rate'])
                        filters_obj = MEPFilters(sampling_rate=data['sampling_rate'])
                        
                        fig, ax = plt.subplots(figsize=(15, 9))
                        
                        # Plot ground truth with highest z-order
                        ax.plot(data['time']*1000, data['mep_clean'], 'k-', 
                               linewidth=3.5, alpha=1.0, label='Ground Truth', zorder=10000)
                        ax.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1.5, label='TMS Stimulus')
                        
                        # Color map for different configurations
                        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_configs)))
                        
                        # Plot each selected configuration (reverse order for z-ordering)
                        for config_idx, (config_name, color) in enumerate(reversed(list(zip(selected_configs, colors)))):
                            config_data = df[df['config_label'] == config_name]
                            
                            if len(config_data) == 0:
                                continue
                            
                            # Calculate z-order (first selected = top layer)
                            z_base = 1000 - (config_idx * 100)
                            
                            # Get filter parameters
                            first_row = config_data.iloc[0]
                            filter_params = {
                                'filter_type': first_row['filter_type'],
                                'lowcut': first_row['lowcut'],
                                'highcut': first_row['highcut'],
                                'order': first_row['order'],
                                'notch_enabled': first_row.get('notch_enabled', False),
                                'notch_freq': 50
                            }
                            
                            # Collect all filtered signals for this config
                            all_filtered = []
                            
                            for idx, row in config_data.iterrows():
                                # Generate noisy signal
                                mep_noisy = noise_gen.add_composite_noise(
                                    data['mep_clean'], data['time'],
                                    snr_db=data['parameters']['snr_db'],
                                    include_line=True,
                                    include_emg=True
                                )
                                
                                # Apply filter
                                try:
                                    mep_filtered = filters_obj.apply_filter_cascade(mep_noisy, filter_params)
                                    all_filtered.append(mep_filtered)
                                    
                                    # Plot individual iterations if requested
                                    if show_individual and not show_mean_only:
                                        ax.plot(data['time']*1000, mep_filtered, 
                                               color=color, linewidth=0.4, 
                                               alpha=alpha_iterations, zorder=z_base - 50)
                                except:
                                    pass
                            
                            # Calculate and plot mean
                            if len(all_filtered) > 0:
                                all_filtered = np.array(all_filtered)
                                mean_filtered = np.mean(all_filtered, axis=0)
                                std_filtered = np.std(all_filtered, axis=0)
                                
                                # Plot SD envelope first (lower z-order)
                                if show_sd_envelope:
                                    ax.fill_between(data['time']*1000,
                                                   mean_filtered - std_filtered,
                                                   mean_filtered + std_filtered,
                                                   color=color, alpha=0.2, zorder=z_base - 25,
                                                   label=f'{config_name} ±SD')
                                
                                # Plot mean with higher z-order (on top of envelope)
                                ax.plot(data['time']*1000, mean_filtered, 
                                       color=color, linewidth=2.8, alpha=0.95,
                                       label=f'{config_name} (n={len(all_filtered)})',
                                       zorder=z_base)
                        
                        ax.set_xlabel('Time (ms) - Negative = Pre-Stimulus', fontsize=13, fontweight='bold')
                        ax.set_ylabel('Amplitude (mV)', fontsize=13, fontweight='bold')
                        ax.set_title('Multi-Filter Performance Overlay', fontsize=15, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
                        ax.grid(True, alpha=0.2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add download button
                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=overlay_dpi, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label=f"💾 Download Multi-Filter Overlay ({overlay_dpi} DPI)",
                            data=buf,
                            file_name=f"multi_filter_overlay_{overlay_dpi}dpi.png",
                            mime="image/png",
                            key='download_overlay'
                        )
                        
                        plt.close()
                        
                        st.success(f"✅ Overlaid {len(selected_configs)} filter configurations with proper z-ordering")
                        st.info("**Z-ordering:** Configurations are layered with the first selected on top for best visibility")
            
            # TAB 3: Heatmap
            with viz_tab3:
                col_heat1, col_heat2 = st.columns([3, 1])
                
                with col_heat1:
                    metric_choice_heat = st.selectbox(
                        "Select metric for heatmap",
                        ['amplitude_error_pct', 'peak_latency_error_ms', 
                         'correlation', 'rmse_mep', 'baseline_std'],
                        key='heatmap_metric'
                    )
                
                with col_heat2:
                    st.write("**Export Options:**")
                    heatmap_dpi = st.selectbox("DPI", [150, 300, 600], index=1, key='heatmap_dpi')
                
                # Heatmap generation (keeping existing code)
                # For amplitude error, use absolute values for sorting and display
                if metric_choice_heat == 'amplitude_error_pct':
                    # Calculate absolute error for sorting
                    pivot_data = df.groupby('config_label')[metric_choice_heat].mean()
                    pivot_data_abs = pivot_data.abs().sort_values()
                    pivot_data = pivot_data.loc[pivot_data_abs.index]  # Reorder by absolute value
                    
                    # Create custom colormap centered at 0
                    fig, ax = plt.subplots(figsize=(10, len(pivot_data)*0.3 + 2))
                    
                    # Use diverging colormap centered at 0
                    vmax = max(abs(pivot_data.min()), abs(pivot_data.max()))
                    sns.heatmap(pivot_data.to_frame(), annot=True, fmt='.3f', 
                               cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
                               ax=ax, cbar_kws={'label': metric_choice_heat})
                    ax.set_xlabel('')
                    ax.set_ylabel('Filter Configuration (sorted by |error|)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add download button
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=heatmap_dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label=f"💾 Download Heatmap ({heatmap_dpi} DPI)",
                        data=buf,
                        file_name=f"heatmap_{metric_choice_heat}_{heatmap_dpi}dpi.png",
                        mime="image/png",
                        key='download_heatmap_amp'
                    )
                    
                    plt.close()
                else:
                    # Standard sorting for other metrics
                    pivot_data = df.groupby('config_label')[metric_choice_heat].mean().sort_values()
                    
                    fig, ax = plt.subplots(figsize=(10, len(pivot_data)*0.3 + 2))
                    
                    # Determine if higher or lower is better
                    if metric_choice_heat in ['correlation']:
                        cmap = 'RdYlGn'  # Higher is better (green)
                    else:
                        cmap = 'RdYlGn_r'  # Lower is better (green)
                    
                    sns.heatmap(pivot_data.to_frame(), annot=True, fmt='.3f', 
                               cmap=cmap, ax=ax, cbar_kws={'label': metric_choice_heat})
                    ax.set_xlabel('')
                    ax.set_ylabel('Filter Configuration')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add download button
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=heatmap_dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label=f"💾 Download Heatmap ({heatmap_dpi} DPI)",
                        data=buf,
                        file_name=f"heatmap_{metric_choice_heat}_{heatmap_dpi}dpi.png",
                        mime="image/png",
                        key='download_heatmap_other'
                    )
                    
                    plt.close()
            
            # Best configurations
            st.subheader("🏆 Best Performing Filters")
            st.write("**Based on mean performance across all iterations**")
            
            # Calculate mean metrics per configuration for best selection
            config_means = df.groupby('config_label').agg({
                'amplitude_error_pct': ['mean', 'std'],
                'peak_latency_error_ms': ['mean', 'std'],
                'correlation': ['mean', 'std']
            })
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Lowest Amplitude Error**")
                best_amp_idx = config_means[('amplitude_error_pct', 'mean')].abs().idxmin()
                best_amp_mean = config_means.loc[best_amp_idx, ('amplitude_error_pct', 'mean')]
                best_amp_std = config_means.loc[best_amp_idx, ('amplitude_error_pct', 'std')]
                st.write(f"Config: {best_amp_idx}")
                st.write(f"Error: {best_amp_mean:.2f}% ± {best_amp_std:.2f}%")
            
            with col2:
                st.write("**Lowest Latency Error**")
                best_lat_idx = config_means[('peak_latency_error_ms', 'mean')].abs().idxmin()
                best_lat_mean = config_means.loc[best_lat_idx, ('peak_latency_error_ms', 'mean')]
                best_lat_std = config_means.loc[best_lat_idx, ('peak_latency_error_ms', 'std')]
                st.write(f"Config: {best_lat_idx}")
                st.write(f"Error: {best_lat_mean:.3f} ± {best_lat_std:.3f} ms")
            
            with col3:
                st.write("**Highest Correlation**")
                best_corr_idx = config_means[('correlation', 'mean')].idxmax()
                best_corr_mean = config_means.loc[best_corr_idx, ('correlation', 'mean')]
                best_corr_std = config_means.loc[best_corr_idx, ('correlation', 'std')]
                st.write(f"Config: {best_corr_idx}")
                st.write(f"Correlation: {best_corr_mean:.4f} ± {best_corr_std:.4f}")
            
            # Iteration overlay visualization
            st.subheader("📊 Filter Performance Visualization")
            st.write("**Compare filtered signals across iterations with ground truth**")
            
            # Select configuration to visualize
            config_to_plot = st.selectbox(
                "Select filter configuration to visualize:",
                options=df['config_label'].unique(),
                index=0
            )
            
            # Add DPI selector
            single_overlay_dpi = st.selectbox("Export DPI", [150, 300, 600], index=1, key='single_overlay_dpi')
            
            if st.button("🎨 Generate Overlay Plot"):
                with st.spinner("Generating visualization..."):
                    # Initialize generators and filters for this visualization
                    noise_gen = NoiseGenerator(sampling_rate=data['sampling_rate'])
                    filters = MEPFilters(sampling_rate=data['sampling_rate'])
                    
                    # Get all iterations for this config
                    config_data = df[df['config_label'] == config_to_plot]
                    
                    # Re-apply filter for each iteration to get waveforms
                    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                    
                    # Get filter parameters from first row
                    first_row = config_data.iloc[0]
                    filter_params = {
                        'filter_type': first_row['filter_type'],
                        'lowcut': first_row['lowcut'],
                        'highcut': first_row['highcut'],
                        'order': first_row['order'],
                        'notch_enabled': first_row.get('notch_enabled', False),
                        'notch_freq': 50
                    }
                    
                    # Plot ground truth (bold)
                    axes[0].plot(data['time']*1000, data['mep_clean'], 'b-', 
                               linewidth=2.5, alpha=1.0, label='Ground Truth', zorder=100)
                    
                    # Apply filter and plot each iteration (thin lines)
                    for idx, row in config_data.iterrows():
                        # Generate noisy signal for this iteration
                        mep_noisy = noise_gen.add_composite_noise(
                            data['mep_clean'], data['time'],
                            snr_db=data['parameters']['snr_db'],
                            include_line=True,
                            include_emg=True
                        )
                        
                        # Apply filter
                        try:
                            mep_filtered = filters.apply_filter_cascade(mep_noisy, filter_params)
                            
                            # Plot filtered iteration (thin, semi-transparent)
                            axes[0].plot(data['time']*1000, mep_filtered, 'r-', 
                                       linewidth=0.5, alpha=0.3, zorder=1)
                        except:
                            pass  # Skip if filter fails
                    
                    # Add single legend entry for iterations
                    axes[0].plot([], [], 'r-', linewidth=0.5, alpha=0.5, label=f'Filtered (n={len(config_data)})')
                    
                    # Add stimulus marker
                    axes[0].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1.5, label='TMS Stimulus')
                    
                    axes[0].set_ylabel('Amplitude (mV)', fontsize=12)
                    axes[0].set_title(f'Overlay: {config_to_plot}', fontsize=14, fontweight='bold')
                    axes[0].legend(loc='upper right', fontsize=11)
                    axes[0].grid(True, alpha=0.3)
                    
                    # Bottom plot: Mean ± SD envelope
                    # Calculate mean and std across iterations
                    all_filtered = []
                    for idx, row in config_data.iterrows():
                        mep_noisy = noise_gen.add_composite_noise(
                            data['mep_clean'], data['time'],
                            snr_db=data['parameters']['snr_db'],
                            include_line=True,
                            include_emg=True
                        )
                        try:
                            mep_filtered = filters.apply_filter_cascade(mep_noisy, filter_params)
                            all_filtered.append(mep_filtered)
                        except:
                            pass
                    
                    if len(all_filtered) > 0:
                        all_filtered = np.array(all_filtered)
                        mean_filtered = np.mean(all_filtered, axis=0)
                        std_filtered = np.std(all_filtered, axis=0)
                        
                        # Plot mean with shaded standard deviation
                        axes[1].plot(data['time']*1000, data['mep_clean'], 'b-', 
                                   linewidth=2.5, alpha=0.8, label='Ground Truth')
                        axes[1].plot(data['time']*1000, mean_filtered, 'r-', 
                                   linewidth=2, alpha=0.8, label='Mean Filtered')
                        axes[1].fill_between(data['time']*1000, 
                                           mean_filtered - std_filtered,
                                           mean_filtered + std_filtered,
                                           color='r', alpha=0.2, label='±1 SD')
                        axes[1].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1.5)
                        
                        axes[1].set_xlabel('Time (ms) - Negative = Pre-Stimulus', fontsize=12)
                        axes[1].set_ylabel('Amplitude (mV)', fontsize=12)
                        axes[1].set_title('Mean ± Standard Deviation', fontsize=14, fontweight='bold')
                        axes[1].legend(loc='upper right', fontsize=11)
                        axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add download button
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=single_overlay_dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label=f"💾 Download Overlay Plot ({single_overlay_dpi} DPI)",
                        data=buf,
                        file_name=f"overlay_{config_to_plot}_{single_overlay_dpi}dpi.png",
                        mime="image/png",
                        key='download_single_overlay'
                    )
                    
                    plt.close()
                    
                    # Show statistics
                    st.info(f"""
                    **Visualisation Statistics:**
                    - Configuration: {config_to_plot}
                    - Iterations plotted: {len(all_filtered)}
                    - Mean amplitude error: {config_data['amplitude_error_pct'].mean():.2f}% (±{config_data['amplitude_error_pct'].std():.2f}%)
                    - Mean latency error: {config_data['peak_latency_error_ms'].mean():.3f} ms (±{config_data['peak_latency_error_ms'].std():.3f} ms)
                    - Mean correlation: {config_data['correlation'].mean():.4f} (±{config_data['correlation'].std():.4f})
                    """)
            
            # Download results
            st.subheader("📥 Export Results")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="mep_filter_batch_results.csv",
                mime="text/csv"
            )

# ===========================
# TAB 4: METHODS
# ===========================
with tab4:
    st.header("📋 Methods Section Generator")
    
    st.markdown("""
    This tab generates publication-ready methods text based on your current analysis settings.
    The text adapts to your chosen parameters, making it easy to document your methodology.
    """)
    
    # Generate methods text based on current settings
    methods_parts = []
    references_used = []
    
    # Check what settings are available
    if st.session_state.generated_signals is not None:
        data = st.session_state.generated_signals
        params = data['parameters']
        
        st.subheader("Current Analysis Configuration")
        
        # Display current settings
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MEP Type", params.get('mep_type', 'Not set'))
            st.metric("Amplitude", f"{params.get('amplitude', 0):.2f} mV")
            st.metric("Duration", f"{params.get('duration', 0):.0f} ms")
        with col2:
            st.metric("Sampling Rate", f"{data['sampling_rate']} Hz")
            st.metric("SNR", f"{params.get('snr_db', 0):.0f} dB")
            st.metric("Onset Latency", f"{params.get('onset_latency', 0):.0f} ms")
        with col3:
            if st.session_state.batch_results is not None:
                n_configs = len(st.session_state.batch_results['config_label'].unique())
                n_iters = len(st.session_state.batch_results) // n_configs if n_configs > 0 else 0
                st.metric("Configurations Tested", n_configs)
                st.metric("Iterations per Config", n_iters)
                st.metric("Total Tests", len(st.session_state.batch_results))
            else:
                st.metric("Batch Analysis", "Not run")
        
        st.divider()
        
        # Generate signal methods
        st.subheader("Generated Methods Text")
        
        methods_parts.append("### Signal Generation\n\n")
        
        # MEP morphology description
        mep_type = params.get('mep_type', 'Standard')
        if mep_type == "Standard":
            morph_desc = "Monophasic motor evoked potentials"
        elif mep_type == "Bi-phasic":
            morph_desc = "Bi-phasic motor evoked potentials (positive-negative morphology with 80% phase ratio)"
            references_used.append('groppa2012')
        elif mep_type == "Tri-phasic":
            morph_desc = "Tri-phasic motor evoked potentials (positive-negative-positive morphology with 75% and 40% phase ratios)"
            references_used.append('groppa2012')
        elif mep_type == "Double Peak":
            morph_desc = "Double-peak motor evoked potentials simulating dual-coil stimulation"
        else:
            morph_desc = "Motor evoked potentials with baseline electromyographic activity"
        
        methods_parts.append(f"{morph_desc} were simulated using the MEP Filter Testing Tool v1.4.0 (Andrushko, 2024) ")
        methods_parts.append(f"with amplitude {params['amplitude']:.1f} mV and duration {params['duration']:.0f} ms, ")
        methods_parts.append(f"sampled at {data['sampling_rate']} Hz. ")
        
        # Noise description
        noise_types = params.get('noise_types', {})
        if noise_types:
            methods_parts.append(f"Realistic physiological noise (signal-to-noise ratio: {params['snr_db']:.0f} dB) was added to simulate recording conditions, including ")
            
            noise_desc = []
            if noise_types.get('emg'):
                noise_desc.append("electromyographic activity (20-500 Hz)")
            if noise_types.get('line'):
                noise_desc.append("50 Hz line noise with harmonics")
            if noise_types.get('ecg'):
                noise_desc.append("electrocardiographic artefacts")
            if noise_types.get('movement'):
                noise_desc.append("low-frequency movement artefacts")
            if noise_types.get('tms'):
                noise_desc.append("residual transcranial magnetic stimulation artefact (1.5 mV amplitude, 2 ms exponential decay)")
                references_used.append('rossini2015')
            
            if noise_desc:
                methods_parts.append(", ".join(noise_desc) + ". ")
        
        # Filter analysis
        if st.session_state.batch_results is not None:
            df = st.session_state.batch_results
            n_configs = len(df['config_label'].unique())
            n_iterations = len(df) // n_configs if n_configs > 0 else 0
            
            methods_parts.append("\n\n### Digital Filter Evaluation\n\n")
            methods_parts.append(f"Digital filters were evaluated systematically across {n_configs} configurations ")
            
            # Describe filter types
            filter_types = df['filter_type'].unique()
            filter_desc = []
            if any('butterw' in f for f in filter_types):
                orders = sorted(df[df['filter_type'].str.contains('butterw')]['order'].unique())
                filter_desc.append(f"Butterworth (orders: {', '.join(map(str, orders))})")
                references_used.append('butterworth')
            if any('fir' in f for f in filter_types):
                fir_types = [f.replace('fir_', '').capitalize() for f in filter_types if 'fir' in f]
                filter_desc.append(f"FIR ({', '.join(set(fir_types))} windows)")
                references_used.append('fir')
            
            methods_parts.append(f"spanning {' and '.join(filter_desc)} implementations ")
            
            # Describe frequency bands
            hp_cutoffs = sorted(df['lowcut'].unique())
            lp_cutoffs = sorted(df['highcut'].unique())
            methods_parts.append(f"with highpass cutoffs of {', '.join(map(str, [int(x) for x in hp_cutoffs]))} Hz ")
            methods_parts.append(f"and lowpass cutoffs of {', '.join(map(str, [int(x) for x in lp_cutoffs]))} Hz. ")
            
            # Notch filter
            if df['notch_enabled'].any():
                methods_parts.append("Selected configurations included a 50 Hz notch filter for line noise rejection. ")
            
            methods_parts.append(f"Each configuration was tested across {n_iterations} noise realisations to assess robustness. ")
            
            methods_parts.append("\n\n### Performance Metrics\n\n")
            methods_parts.append("Filter performance was quantified using: ")
            methods_parts.append("(1) amplitude preservation error (percentage deviation from ground truth peak-to-peak amplitude); ")
            methods_parts.append("(2) peak latency shift (temporal displacement of maximum amplitude in milliseconds); ")
            methods_parts.append("(3) Pearson correlation coefficient between filtered and ground truth waveforms; ")
            methods_parts.append("and (4) root mean square error across the motor evoked potential window. ")
            methods_parts.append("All metrics are reported as mean ± standard deviation across iterations. ")
            references_used.append('groppa2012')
            references_used.append('rossini2015')
            
            methods_parts.append("\n\n### Statistical Analysis\n\n")
            methods_parts.append(f"Performance was aggregated across {n_iterations} iterations using mean and standard deviation. ")
            methods_parts.append("Optimal filter configurations were identified based on minimal absolute amplitude error, ")
            methods_parts.append("lowest peak latency shift, and highest waveform correlation. ")
        
        # Display generated methods
        methods_text = "".join(methods_parts)
        
        st.markdown(methods_text)
        
        st.divider()
        
        # Download options
        st.subheader("📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Methods Text**")
            # Add tool citation
            full_methods = methods_text + "\n\n### Tool Citation\n\n"
            full_methods += "Andrushko, J.W. (2024). MEP Filter Testing Tool: Systematic Digital Filter Evaluation for Motor Evoked Potentials (Version 1.4.0). Northumbria University. https://github.com/andrushko/mep-filter-tool\n"
            
            st.download_button(
                label="📄 Download Methods as TXT",
                data=full_methods,
                file_name="mep_filter_methods.txt",
                mime="text/plain",
                key='download_methods_txt'
            )
        
        with col2:
            st.write("**References**")
            
            # Generate RIS file
            ris_content = """TY  - JOUR
AU  - Groppa, Sergiu
AU  - Oliviero, Antonio
AU  - Eisen, Andrew
AU  - Quartarone, Angelo
AU  - Cohen, Leonardo G
AU  - Mall, Volker
AU  - Kaelin-Lang, Alain
AU  - Mima, Tatsuya
AU  - Rossi, Simone
AU  - Thickbroom, Gary W
AU  - Rossini, Paolo M
AU  - Ziemann, Ulf
AU  - Valls-Solé, Josep
AU  - Siebner, Hartwig R
TI  - A practical guide to diagnostic transcranial magnetic stimulation: Report of an IFCN committee
JO  - Clinical Neurophysiology
VL  - 123
IS  - 5
SP  - 858
EP  - 882
PY  - 2012
DO  - 10.1016/j.clinph.2012.01.010
ER  -

TY  - JOUR
AU  - Rossini, Paolo M
AU  - Burke, David
AU  - Chen, Robert
AU  - Cohen, Leonardo G
AU  - Daskalakis, Zafiris
AU  - Di Iorio, Riccardo
AU  - Di Lazzaro, Vincenzo
AU  - Ferreri, Florinda
AU  - Fitzgerald, Paul B
AU  - George, Mark S
AU  - Hallett, Mark
AU  - Lefaucheur, Jean Pascal
AU  - Langguth, Berthold
AU  - Matsumoto, Hideyuki
AU  - Miniussi, Carlo
AU  - Nitsche, Michael A
AU  - Pascual-Leone, Alvaro
AU  - Paulus, Walter
AU  - Rossi, Simone
AU  - Rothwell, John C
AU  - Siebner, Hartwig R
AU  - Ugawa, Yoshikazu
AU  - Walsh, Vincent
AU  - Ziemann, Ulf
TI  - Non-invasive electrical and magnetic stimulation of the brain, spinal cord, roots and peripheral nerves: Basic principles and procedures for routine clinical and research application
JO  - Clinical Neurophysiology
VL  - 126
IS  - 6
SP  - 1071
EP  - 1107
PY  - 2015
DO  - 10.1016/j.clinph.2015.02.001
ER  -

TY  - SOFT
AU  - Andrushko, Justin W
TI  - MEP Filter Testing Tool: Systematic Digital Filter Evaluation for Motor Evoked Potentials
PY  - 2024
PB  - Northumbria University
VL  - 1.4.0
UR  - https://github.com/andrushko/mep-filter-tool
ER  -

"""
            
            st.download_button(
                label="📚 Download References (RIS)",
                data=ris_content,
                file_name="mep_filter_references.ris",
                mime="application/x-research-info-systems",
                key='download_ris'
            )
        
        st.divider()
        
        # Reference list
        st.subheader("📚 Key References")
        
        st.markdown("""
        **Essential TMS-EMG Guidelines:**
        
        1. **Groppa, S., et al. (2012).** A practical guide to diagnostic transcranial magnetic stimulation: 
           Report of an IFCN committee. *Clinical Neurophysiology*, 123(5), 858-882. 
           https://doi.org/10.1016/j.clinph.2012.01.010
        
        2. **Rossini, P.M., et al. (2015).** Non-invasive electrical and magnetic stimulation of the brain, 
           spinal cord, roots and peripheral nerves: Basic principles and procedures for routine clinical 
           and research application. *Clinical Neurophysiology*, 126(6), 1071-1107. 
           https://doi.org/10.1016/j.clinph.2015.02.001
        
        **Tool Citation:**
        
        3. **Andrushko, J.W. (2024).** MEP Filter Testing Tool: Systematic Digital Filter Evaluation for 
           Motor Evoked Potentials (Version 1.4.0). Northumbria University. 
           https://github.com/andrushko/mep-filter-tool
        
        **Additional Resources:**
        
        - Digital filter design: Oppenheim, A.V., & Schafer, R.W. (2009). *Discrete-Time Signal Processing* (3rd ed.)
        - EMG analysis: Merletti, R., & Parker, P.A. (2004). *Electromyography: Physiology, Engineering, and Noninvasive Applications*
        - TMS methodology: Hallett, M. (2007). Transcranial magnetic stimulation: A primer. *Neuron*, 55(2), 187-199.
        """)
        
        st.info("""
        **💡 Tip:** Download the methods text and RIS file above. The RIS file can be imported directly into 
        EndNote, Zotero, Mendeley, or other reference managers. Edit the methods text as needed for your 
        specific manuscript.
        """)
    
    else:
        st.warning("⚠️ Generate a signal first to see adaptive methods text based on your settings.")
        st.info("""
        Once you've:
        1. Generated a signal (Tab 1)
        2. Optionally run batch analysis (Tab 3)
        
        This tab will display publication-ready methods text that adapts to your specific parameters.
        """)

# ===========================
# TAB 5: ABOUT
# ===========================
with tab5:
    st.header("About This Tool")
    
    st.markdown("""
    ### MEP Filter Testing Tool
    
    **Version:** 1.4.0 Professional Edition  
    **Author:** Justin W. Andrushko, PhD  
    **Institution:** Northumbria University
    
    ### Purpose
    
    This tool was developed to systematically evaluate digital filter performance for 
    Motor Evoked Potential (MEP) analysis in TMS studies. It addresses the critical need 
    for evidence-based filter selection in neurophysiology research.
    
    ### Features
    
    - **Realistic MEP Simulation**: Generate MEPs with configurable amplitude, duration, 
      rise/fall times, and morphology
    - **Comprehensive Noise Models**: Add physiological (EMG, ECG) and technical 
      (line noise, amplifier noise) artifacts
    - **Multiple Filter Types**: Test Butterworth, FIR (various windows), notch, and 
      smoothing filters
    - **Robust Metrics**: Evaluate amplitude accuracy, timing precision, morphology 
      preservation, and SNR improvement
    - **Batch Analysis**: Automatically test multiple configurations and compare performance
    - **Interactive Visualization**: Real-time plotting and exploration of results
    
    ### Key Metrics
    
    **Amplitude Metrics:**
    - Peak-to-peak amplitude error (absolute and percentage)
    - Area under curve (AUC) error
    
    **Timing Metrics:**
    - MEP onset latency error
    - Peak latency error
    
    **Morphology Metrics:**
    - Correlation with ground truth
    - RMSE (full signal and MEP window)
    - Baseline stability
    
    **Quality Metrics:**
    - Signal-to-noise ratio improvement
    - Spurious oscillation detection
    
    ### Recommended Workflow
    
    1. **Generate Signal**: Start with the Signal Generation tab to create a realistic MEP 
       with desired noise characteristics
    2. **Test Filters**: Use the Filter Testing tab to interactively explore different 
       filter settings
    3. **Batch Analysis**: Once you have candidate filters, run batch analysis to compare 
       them systematically across multiple noise realizations
    4. **Export Results**: Download results for further analysis or inclusion in publications
    
    ### Technical Notes
    
    - **Zero-Phase Filtering**: Most filters use `filtfilt()` to eliminate phase distortion
    - **Edge Effects**: Signals are appropriately padded before filtering
    - **Sampling Rate**: Default 2 kHz (adjustable) matches typical EMG recording systems
    - **Onset Detection**: Uses threshold-based method with baseline statistics
    
    ### Citation
    
    If you use this tool in your research, please cite:
    
    ```
    Andrushko, J.W. (2024). MEP Filter Testing Tool: A Systematic Approach to 
    Digital Filter Selection for Motor Evoked Potentials. Northumbria University.
    ```
    
    ### Feedback & Contributions
    
    This tool is open for community feedback and contributions. If you have suggestions 
    for improvements, bug reports, or feature requests, please contact the author.
    
    ### Acknowledgments
    
    Developed with support from Northumbria University's Neuromuscular Function research group.
    
    ---
    
    **License:** MIT License  
    **Contact:** justin.andrushko@northumbria.ac.uk
    """)

# Sidebar
with st.sidebar:
    st.header("Quick Actions")
    
    if st.button("🗑️ Clear All Results"):
        st.session_state.filter_results = []
        st.session_state.batch_results = None
        st.rerun()
    
    if st.button("🔄 Reset Everything"):
        st.session_state.generated_signals = None
        st.session_state.filter_results = []
        st.session_state.batch_results = None
        st.rerun()
    
    st.divider()
    
    st.header("Current Status")
    
    if st.session_state.generated_signals is not None:
        st.success("✅ Signal generated")
        st.write(f"**Filters tested:** {len(st.session_state.filter_results)}")
    else:
        st.info("ℹ️ No signal generated")
    
    if st.session_state.batch_results is not None:
        st.success(f"✅ Batch results: {len(st.session_state.batch_results)} tests")
    
    st.divider()
    
    st.header("Resources")
    st.markdown("""
    - [TMS-EMG Best Practices](https://doi.org/10.1016/j.clinph.2014.07.023)
    - [Digital Filter Design](https://scipy.github.io/devdocs/tutorial/signal.html)
    - [MEP Analysis Guidelines](https://doi.org/10.1016/j.brs.2018.04.018)
    """)
