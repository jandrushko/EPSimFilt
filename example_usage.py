"""
Example Script: Using MEP Filter Tool Modules
This demonstrates how to use the tool modules programmatically without the GUI.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signal_generator import MEPGenerator
from noise_generator import NoiseGenerator
from filters import MEPFilters
from metrics import MEPMetrics


def example_basic_workflow():
    """Basic workflow: generate, add noise, filter, evaluate."""
    
    print("=" * 60)
    print("MEP Filter Tool - Basic Workflow Example")
    print("=" * 60)
    
    # 1. Initialize modules
    sampling_rate = 2000  # Hz
    mep_gen = MEPGenerator(sampling_rate=sampling_rate)
    noise_gen = NoiseGenerator(sampling_rate=sampling_rate)
    filters = MEPFilters(sampling_rate=sampling_rate)
    metrics = MEPMetrics(sampling_rate=sampling_rate)
    
    # 2. Generate clean MEP
    print("\n1. Generating clean MEP signal...")
    time, mep_clean = mep_gen.generate_mep(
        amplitude=1.0,      # 1 mV
        duration=0.030,     # 30 ms
        onset_latency=0.020,  # 20 ms
        rise_time=0.008,    # 8 ms
        decay_time=0.015,   # 15 ms
        asymmetry=1.2
    )
    print(f"   Signal duration: {time[-1]*1000:.1f} ms")
    print(f"   Sampling rate: {sampling_rate} Hz")
    print(f"   Number of samples: {len(time)}")
    
    # 3. Add noise
    print("\n2. Adding composite noise (SNR = 10 dB)...")
    mep_noisy = noise_gen.add_composite_noise(
        mep_clean, time,
        snr_db=10.0,
        include_line=True,
        include_emg=True,
        include_ecg=False,
        include_movement=False
    )
    
    # 4. Apply filters
    print("\n3. Applying filters...")
    
    # Butterworth filter (zero-phase)
    filter_params_butter = {
        'filter_type': 'butterworth',
        'lowcut': 10,
        'highcut': 500,
        'order': 4,
        'notch_enabled': True,
        'notch_freq': 50
    }
    mep_butter = filters.apply_filter_cascade(mep_noisy, filter_params_butter)
    print("   ✓ Butterworth 4th-order (10-500 Hz) + 50 Hz notch")
    
    # FIR filter
    filter_params_fir = {
        'filter_type': 'fir_hamming',
        'lowcut': 10,
        'highcut': 500,
        'order': 4,
        'notch_enabled': False
    }
    mep_fir = filters.apply_filter_cascade(mep_noisy, filter_params_fir)
    print("   ✓ FIR Hamming (10-500 Hz)")
    
    # 5. Calculate metrics
    print("\n4. Calculating performance metrics...")
    
    metrics_butter = metrics.calculate_all_metrics(mep_clean, mep_butter, time)
    metrics_fir = metrics.calculate_all_metrics(mep_clean, mep_fir, time)
    
    print("\n   Butterworth Filter:")
    print(f"   - Amplitude error: {metrics_butter['amplitude_error_pct']:.2f}%")
    print(f"   - Peak latency error: {metrics_butter['peak_latency_error_ms']:.3f} ms")
    print(f"   - Correlation: {metrics_butter['correlation']:.4f}")
    print(f"   - RMSE: {metrics_butter['rmse_mep']:.4f} mV")
    
    print("\n   FIR Filter:")
    print(f"   - Amplitude error: {metrics_fir['amplitude_error_pct']:.2f}%")
    print(f"   - Peak latency error: {metrics_fir['peak_latency_error_ms']:.3f} ms")
    print(f"   - Correlation: {metrics_fir['correlation']:.4f}")
    print(f"   - RMSE: {metrics_fir['rmse_mep']:.4f} mV")
    
    # 6. Visualize results
    print("\n5. Generating visualization...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Left column: Waveforms
    axes[0, 0].plot(time*1000, mep_clean, 'b-', linewidth=2, label='Clean')
    axes[0, 0].set_title('Clean MEP Signal')
    axes[0, 0].set_ylabel('Amplitude (mV)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(time*1000, mep_noisy, 'gray', linewidth=0.8, alpha=0.7, label='Noisy')
    axes[1, 0].plot(time*1000, mep_clean, 'b--', linewidth=1, alpha=0.5, label='Clean')
    axes[1, 0].set_title('Noisy Signal (SNR = 10 dB)')
    axes[1, 0].set_ylabel('Amplitude (mV)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(time*1000, mep_clean, 'b-', linewidth=1.5, alpha=0.6, label='Clean')
    axes[2, 0].plot(time*1000, mep_butter, 'r-', linewidth=1.5, alpha=0.8, label='Butterworth')
    axes[2, 0].plot(time*1000, mep_fir, 'g-', linewidth=1.5, alpha=0.8, label='FIR')
    axes[2, 0].set_title('Filtered Signals Comparison')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].set_ylabel('Amplitude (mV)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Right column: Analysis
    # Frequency response
    freqs_butter, resp_butter = filters.get_frequency_response(filter_params_butter)
    freqs_fir, resp_fir = filters.get_frequency_response(filter_params_fir)
    
    axes[0, 1].plot(freqs_butter, resp_butter, 'r-', linewidth=2, label='Butterworth')
    axes[0, 1].plot(freqs_fir, resp_fir, 'g-', linewidth=2, label='FIR')
    axes[0, 1].axhline(-3, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Filter Frequency Response')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_xlim([0, 1000])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error comparison
    error_butter = mep_butter - mep_clean
    error_fir = mep_fir - mep_clean
    
    axes[1, 1].plot(time*1000, error_butter, 'r-', linewidth=1, alpha=0.7, label='Butterworth')
    axes[1, 1].plot(time*1000, error_fir, 'g-', linewidth=1, alpha=0.7, label='FIR')
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Reconstruction Error')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Error (mV)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Metrics comparison bar plot
    metrics_names = ['Amplitude\nError (%)', 'Latency\nError (ms)', 'Correlation', 'RMSE\n(mV)']
    butter_values = [
        abs(metrics_butter['amplitude_error_pct']),
        abs(metrics_butter['peak_latency_error_ms']),
        metrics_butter['correlation'],
        metrics_butter['rmse_mep']
    ]
    fir_values = [
        abs(metrics_fir['amplitude_error_pct']),
        abs(metrics_fir['peak_latency_error_ms']),
        metrics_fir['correlation'],
        metrics_fir['rmse_mep']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # Normalize for visualization (correlation is already 0-1)
    butter_norm = butter_values.copy()
    fir_norm = fir_values.copy()
    butter_norm[2] = 1 - butter_values[2]  # Invert correlation (lower is better for visualization)
    fir_norm[2] = 1 - fir_values[2]
    
    axes[2, 1].bar(x - width/2, butter_norm, width, label='Butterworth', color='red', alpha=0.7)
    axes[2, 1].bar(x + width/2, fir_norm, width, label='FIR', color='green', alpha=0.7)
    axes[2, 1].set_title('Performance Metrics (Lower is Better)')
    axes[2, 1].set_ylabel('Metric Value')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(metrics_names)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mep_filter_example.png', dpi=150, bbox_inches='tight')
    print("   ✓ Figure saved as 'mep_filter_example.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def example_batch_comparison():
    """Example of batch testing multiple filters."""
    
    print("\n" + "=" * 60)
    print("Batch Filter Comparison Example")
    print("=" * 60)
    
    # Initialize
    sampling_rate = 2000
    mep_gen = MEPGenerator(sampling_rate=sampling_rate)
    noise_gen = NoiseGenerator(sampling_rate=sampling_rate)
    filters = MEPFilters(sampling_rate=sampling_rate)
    metrics_calc = MEPMetrics(sampling_rate=sampling_rate)
    
    # Generate clean MEP
    time, mep_clean = mep_gen.generate_mep(amplitude=1.0)
    
    # Test configurations
    configs = [
        {'name': 'Butter-2nd', 'filter_type': 'butterworth', 'lowcut': 10, 'highcut': 500, 'order': 2, 'notch_enabled': False},
        {'name': 'Butter-4th', 'filter_type': 'butterworth', 'lowcut': 10, 'highcut': 500, 'order': 4, 'notch_enabled': False},
        {'name': 'Butter-6th', 'filter_type': 'butterworth', 'lowcut': 10, 'highcut': 500, 'order': 6, 'notch_enabled': False},
        {'name': 'FIR-Hamming', 'filter_type': 'fir_hamming', 'lowcut': 10, 'highcut': 500, 'order': 4, 'notch_enabled': False},
        {'name': 'FIR-Hann', 'filter_type': 'fir_hann', 'lowcut': 10, 'highcut': 500, 'order': 4, 'notch_enabled': False},
    ]
    
    n_iterations = 20
    results = []
    
    print(f"\nTesting {len(configs)} configurations × {n_iterations} iterations...")
    
    for config in configs:
        for i in range(n_iterations):
            # Generate noisy signal
            mep_noisy = noise_gen.add_composite_noise(
                mep_clean, time,
                snr_db=10.0,
                include_line=True,
                include_emg=True
            )
            
            # Apply filter
            mep_filtered = filters.apply_filter_cascade(mep_noisy, config)
            
            # Calculate metrics
            m = metrics_calc.calculate_all_metrics(mep_clean, mep_filtered, time)
            
            results.append({
                'config': config['name'],
                'amplitude_error': abs(m['amplitude_error_pct']),
                'latency_error': abs(m['peak_latency_error_ms']),
                'correlation': m['correlation'],
                'rmse': m['rmse_mep']
            })
    
    # Aggregate results
    print("\nResults (Mean ± SD):")
    print("-" * 60)
    
    for config in configs:
        config_results = [r for r in results if r['config'] == config['name']]
        
        amp_err = [r['amplitude_error'] for r in config_results]
        lat_err = [r['latency_error'] for r in config_results]
        corr = [r['correlation'] for r in config_results]
        rmse = [r['rmse'] for r in config_results]
        
        print(f"\n{config['name']}:")
        print(f"  Amplitude Error: {np.mean(amp_err):.2f} ± {np.std(amp_err):.2f}%")
        print(f"  Latency Error:   {np.mean(lat_err):.3f} ± {np.std(lat_err):.3f} ms")
        print(f"  Correlation:     {np.mean(corr):.4f} ± {np.std(corr):.4f}")
        print(f"  RMSE:            {np.mean(rmse):.4f} ± {np.std(rmse):.4f} mV")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run basic example
    example_basic_workflow()
    
    # Optionally run batch comparison
    run_batch = input("\nRun batch comparison example? (y/n): ")
    if run_batch.lower() == 'y':
        example_batch_comparison()
    
    print("\n✅ All examples completed!")
    print("\nTo use the GUI tool, run: streamlit run app.py")
