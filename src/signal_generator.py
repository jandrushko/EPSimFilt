"""
EP Signal Generator
Generate realistic Motor Evoked Potential (EP) waveforms with configurable parameters.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal as scipy_signal


class EPGenerator:
    """Generate realistic EP waveforms."""
    
    def __init__(self, sampling_rate: int = 2000):
        """
        Initialize EP generator.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz (default: 2000 Hz)
        """
        self.sampling_rate = sampling_rate
    
    def apply_EP_variability(self,
                             base_amplitude: float,
                             base_latency: float,
                             base_duration: float,
                             amplitude_variability: float = 0.0,
                             latency_variability: float = 0.0,
                             duration_variability: float = 0.0) -> Tuple[float, float, float]:
        """
        Apply random variability to EP parameters for more realistic simulations.
        
        Parameters:
        -----------
        base_amplitude : float
            Base amplitude in mV
        base_latency : float
            Base onset latency in seconds
        base_duration : float
            Base duration in seconds
        amplitude_variability : float
            Variability as proportion (e.g., 0.3 = Â±30%)
        latency_variability : float
            Variability in seconds (e.g., 0.003 = Â±3 ms)
        duration_variability : float
            Variability as proportion (e.g., 0.1 = Â±10%)
            
        Returns:
        --------
        amplitude : float
            Varied amplitude
        latency : float
            Varied latency
        duration : float
            Varied duration
        """
        # Apply amplitude variability (multiplicative, ensures positive)
        if amplitude_variability > 0:
            amplitude = base_amplitude * (1.0 + np.random.uniform(-amplitude_variability, amplitude_variability))
        else:
            amplitude = base_amplitude
        
        # Apply latency variability (additive)
        if latency_variability > 0:
            latency = base_latency + np.random.uniform(-latency_variability, latency_variability)
            latency = max(0.010, latency)  # Minimum 10 ms latency
        else:
            latency = base_latency
        
        # Apply duration variability (multiplicative)
        if duration_variability > 0:
            duration = base_duration * (1.0 + np.random.uniform(-duration_variability, duration_variability))
            duration = max(0.005, duration)  # Minimum 5 ms duration
        else:
            duration = base_duration
            
        return amplitude, latency, duration
        
    def generate_EP(self, 
                     amplitude: float = 1.0,
                     duration: float = 0.03,
                     onset_latency: float = 0.02,
                     rise_time: float = 0.008,
                     decay_time: float = 0.015,
                     total_duration: float = 0.1,
                     asymmetry: float = 1.2,
                     pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a realistic EP waveform.
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude in mV (default: 1.0 mV)
        duration : float
            Total EP duration in seconds (default: 30 ms)
        onset_latency : float
            Time from stimulus to EP onset in seconds (default: 20 ms)
        rise_time : float
            Time to reach peak in seconds (default: 8 ms)
        decay_time : float
            Time to return to baseline in seconds (default: 15 ms)
        total_duration : float
            Total signal duration in seconds AFTER stimulus (default: 100 ms)
        asymmetry : float
            Asymmetry factor (>1 = slower decay, default: 1.2)
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms for TMS artifact visibility)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        EP : np.ndarray
            EP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize EP signal
        EP = np.zeros(n_samples)
        
        # Find onset sample (at specified latency after t=0 stimulus)
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate biphasic waveform using gamma distributions
        # This creates a realistic EP shape with asymmetric rise and fall
        
        # Calculate parameters for realistic shape
        t_EP = time[onset_sample:] - onset_latency
        
        # Positive phase (main deflection)
        peak_time = rise_time
        k_rise = 3.0  # Shape parameter
        theta_rise = rise_time / k_rise
        
        positive_phase = (t_EP / theta_rise) ** k_rise * np.exp(-k_rise * t_EP / rise_time)
        positive_phase = positive_phase / np.max(positive_phase) * amplitude
        
        # Negative phase (smaller, slower)
        k_decay = 2.0
        theta_decay = decay_time / k_decay
        t_shifted = t_EP - rise_time
        t_shifted[t_shifted < 0] = 0
        
        negative_phase = -(t_shifted / theta_decay) ** k_decay * np.exp(-k_decay * t_shifted / (decay_time * asymmetry))
        negative_phase = negative_phase / np.abs(np.min(negative_phase)) * amplitude * 0.3  # 30% of positive
        
        # Combine phases
        EP_signal = positive_phase + negative_phase
        
        # Apply envelope to ensure it returns to baseline
        envelope = np.exp(-t_EP / (duration * 0.8))
        EP_signal = EP_signal * envelope
        
        # Insert into full signal
        EP[onset_sample:onset_sample + len(EP_signal)] = EP_signal
        
        return time, EP
    
    def generate_template_based_EP(self,
                                   template: np.ndarray,
                                   amplitude: float = 1.0,
                                   onset_latency: float = 0.02,
                                   total_duration: float = 0.1,
                                   pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate EP based on a real EP template.
        
        Parameters:
        -----------
        template : np.ndarray
            Template EP waveform (will be normalized)
        amplitude : float
            Desired amplitude in mV
        onset_latency : float
            Time from stimulus to EP onset in seconds
        total_duration : float
            Total signal duration AFTER stimulus in seconds
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        EP : np.ndarray
            EP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Normalize template
        template_normalized = template / np.max(np.abs(template)) * amplitude
        
        # Initialize EP signal
        EP = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Insert template
        template_len = len(template_normalized)
        if onset_sample + template_len <= n_samples:
            EP[onset_sample:onset_sample + template_len] = template_normalized
        else:
            EP[onset_sample:] = template_normalized[:n_samples - onset_sample]
            
        return time, EP
    
    def generate_biphasic_advanced(self,
                                   phase1_amplitude: float = 1.0,
                                   phase1_rise_time: float = 0.010,
                                   phase1_decay_time: float = 0.012,
                                   phase2_amplitude_ratio: float = 0.8,
                                   phase2_rise_time: float = 0.008,
                                   phase2_decay_time: float = 0.010,
                                   phase_separation: float = 0.003,
                                   onset_latency: float = 0.02,
                                   total_duration: float = 0.1,
                                   pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bi-phasic EP with independent control over each phase.
        
        Parameters:
        -----------
        phase1_amplitude : float
            Phase 1 peak amplitude in mV (positive deflection)
        phase1_rise_time : float
            Phase 1 rise time in seconds
        phase1_decay_time : float
            Phase 1 decay time in seconds
        phase2_amplitude_ratio : float
            Phase 2 amplitude as ratio of Phase 1 (e.g., 0.8 = 80% of Phase 1)
        phase2_rise_time : float
            Phase 2 rise time in seconds
        phase2_decay_time : float
            Phase 2 decay time in seconds  
        phase_separation : float
            Time between phase peaks in seconds (default: 3 ms)
        onset_latency : float
            Time from stimulus to EP onset in seconds
        total_duration : float
            Total signal duration AFTER stimulus in seconds
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative = pre-stimulus)
        EP : np.ndarray
            Bi-phasic EP waveform in mV
        """
        # Create time vector
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize signal
        EP = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # PHASE 1: Positive deflection
        phase1_duration = phase1_rise_time + phase1_decay_time
        n_phase1 = int(phase1_duration * self.sampling_rate)
        t_phase1 = np.linspace(0, phase1_duration, n_phase1)
        
        # Gaussian peak for phase 1
        peak1_time = phase1_rise_time
        sigma1 = phase1_rise_time / 2.5  # Adjust for desired width
        phase1 = phase1_amplitude * np.exp(-((t_phase1 - peak1_time)**2) / (2 * sigma1**2))
        
        # Apply smooth onset ramp (3 ms sigmoid)
        ramp_duration = min(0.003, phase1_rise_time * 0.3)
        ramp_samples = int(ramp_duration * self.sampling_rate)
        if ramp_samples > 0 and ramp_samples < len(phase1):
            ramp = 0.5 * (1 + np.tanh((np.linspace(0, 3, ramp_samples) - 1.5) / 0.5))
            phase1[:ramp_samples] *= ramp
        
        # Insert Phase 1
        end_idx1 = min(onset_sample + n_phase1, n_samples)
        EP[onset_sample:end_idx1] = phase1[:end_idx1 - onset_sample]
        
        # PHASE 2: Negative deflection
        phase2_amplitude = phase1_amplitude * phase2_amplitude_ratio
        phase2_onset = onset_latency + phase1_rise_time + phase_separation
        phase2_onset_sample = np.argmin(np.abs(time - phase2_onset))
        
        phase2_duration = phase2_rise_time + phase2_decay_time
        n_phase2 = int(phase2_duration * self.sampling_rate)
        t_phase2 = np.linspace(0, phase2_duration, n_phase2)
        
        # Gaussian peak for phase 2 (negative)
        peak2_time = phase2_rise_time
        sigma2 = phase2_rise_time / 2.5
        phase2 = -phase2_amplitude * np.exp(-((t_phase2 - peak2_time)**2) / (2 * sigma2**2))
        
        # Insert Phase 2
        end_idx2 = min(phase2_onset_sample + n_phase2, n_samples)
        EP[phase2_onset_sample:end_idx2] += phase2[:end_idx2 - phase2_onset_sample]
        
        # REVIEWER FIX: Apply smoothing for perfectly smooth waveforms
        from scipy.signal import savgol_filter
        window_length = min(101, max(51, len(EP) // 15))
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 5:
            EP = savgol_filter(EP, window_length, polyorder=3)
        
        return time, EP
    
    def generate_triphasic_advanced(self,
                                    phase1_amplitude: float = 1.0,
                                    phase1_rise_time: float = 0.010,
                                    phase1_decay_time: float = 0.012,
                                    phase2_amplitude_ratio: float = 0.75,
                                    phase2_rise_time: float = 0.008,
                                    phase2_decay_time: float = 0.010,
                                    phase3_amplitude_ratio: float = 0.40,
                                    phase3_rise_time: float = 0.007,
                                    phase3_decay_time: float = 0.009,
                                    phase1_2_separation: float = 0.003,
                                    phase2_3_separation: float = 0.003,
                                    onset_latency: float = 0.02,
                                    total_duration: float = 0.1,
                                    pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate tri-phasic EP with independent control over each phase.
        
        Parameters:
        -----------
        phase1_amplitude : float
            Phase 1 amplitude in mV (reference, 100%)
        phase1_rise_time : float
            Phase 1 rise time in seconds
        phase1_decay_time : float
            Phase 1 decay time in seconds
        phase2_amplitude_ratio : float
            Phase 2 amplitude ratio (e.g., 0.75 = 75% of Phase 1)
        phase2_rise_time : float
            Phase 2 rise time in seconds
        phase2_decay_time : float
            Phase 2 decay time in seconds
        phase3_amplitude_ratio : float
            Phase 3 amplitude ratio (e.g., 0.40 = 40% of Phase 1)
        phase3_rise_time : float
            Phase 3 rise time in seconds
        phase3_decay_time : float
            Phase 3 decay time in seconds
        phase1_2_separation : float
            Time between Phase 1 and 2 peaks (seconds)
        phase2_3_separation : float
            Time between Phase 2 and 3 peaks (seconds)
        onset_latency : float
            Time from stimulus to EP onset (seconds)
        total_duration : float
            Total signal duration AFTER stimulus (seconds)
        pre_stim_duration : float
            Pre-stimulus baseline duration (seconds)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds
        EP : np.ndarray
            Tri-phasic EP waveform in mV
        """
        # Create time vector
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize signal
        EP = np.zeros(n_samples)
        
        # Helper function to create Gaussian phase
        def create_phase(amplitude, rise_time, decay_time, polarity=1):
            duration = rise_time + decay_time
            n_samp = int(duration * self.sampling_rate)
            t = np.linspace(0, duration, n_samp)
            peak_time = rise_time
            sigma = rise_time / 2.5
            phase = polarity * amplitude * np.exp(-((t - peak_time)**2) / (2 * sigma**2))
            
            # Smooth onset with more pronounced pre-dip
            ramp_dur = min(0.005, rise_time * 0.5)  # Longer ramp
            ramp_samp = int(ramp_dur * self.sampling_rate)
            if ramp_samp > 0 and ramp_samp < len(phase):
                # Shallower slope for deeper undershoot
                ramp = 0.5 * (1 + np.tanh((np.linspace(0, 3, ramp_samp) - 1.5) / 0.8))
                phase[:ramp_samp] *= ramp
            
            return phase
        
        # PHASE 1: Positive
        onset_sample = np.argmin(np.abs(time - onset_latency))
        phase1 = create_phase(phase1_amplitude, phase1_rise_time, phase1_decay_time, polarity=1)
        end_idx1 = min(onset_sample + len(phase1), n_samples)
        EP[onset_sample:end_idx1] = phase1[:end_idx1 - onset_sample]
        
        # PHASE 2: Negative
        phase2_onset = onset_latency + phase1_rise_time + phase1_2_separation
        phase2_onset_sample = np.argmin(np.abs(time - phase2_onset))
        phase2_amp = phase1_amplitude * phase2_amplitude_ratio
        phase2 = create_phase(phase2_amp, phase2_rise_time, phase2_decay_time, polarity=-1)
        end_idx2 = min(phase2_onset_sample + len(phase2), n_samples)
        EP[phase2_onset_sample:end_idx2] += phase2[:end_idx2 - phase2_onset_sample]
        
        # PHASE 3: Positive
        phase3_onset = phase2_onset + phase2_rise_time + phase2_3_separation
        phase3_onset_sample = np.argmin(np.abs(time - phase3_onset))
        phase3_amp = phase1_amplitude * phase3_amplitude_ratio
        phase3 = create_phase(phase3_amp, phase3_rise_time, phase3_decay_time, polarity=1)
        end_idx3 = min(phase3_onset_sample + len(phase3), n_samples)
        EP[phase3_onset_sample:end_idx3] += phase3[:end_idx3 - phase3_onset_sample]
        
        # REVIEWER FIX: Apply smoothing for perfectly smooth waveforms
        from scipy.signal import savgol_filter
        window_length = min(101, max(51, len(EP) // 15))
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 5:
            EP = savgol_filter(EP, window_length, polyorder=3)
        
        return time, EP
    
    def generate_double_peak_EP(self,
                                amplitude1: float = 1.0,
                                amplitude2: float = 0.6,
                                peak_separation: float = 0.005,
                                onset_latency: float = 0.02,
                                total_duration: float = 0.1,
                                pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate EP with double peak (occasionally seen with dual coil stimulation).
        
        Parameters:
        -----------
        amplitude1 : float
            First peak amplitude in mV
        amplitude2 : float
            Second peak amplitude in mV
        peak_separation : float
            Time between peaks in seconds (default: 5 ms)
        onset_latency : float
            Time from stimulus to first EP onset in seconds
        total_duration : float
            Total signal duration AFTER stimulus in seconds
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        EP : np.ndarray
            EP waveform in mV
        """
        # Generate first EP
        time, EP1 = self.generate_EP(amplitude=amplitude1, 
                                       onset_latency=onset_latency,
                                       total_duration=total_duration,
                                       pre_stim_duration=pre_stim_duration)
        
        # Generate second EP with offset
        _, EP2 = self.generate_EP(amplitude=amplitude2,
                                    onset_latency=onset_latency + peak_separation,
                                    total_duration=total_duration,
                                    pre_stim_duration=pre_stim_duration)
        
        # Combine
        EP = EP1 + EP2
        
        return time, EP
    
    def add_baseline_emg(self,
                        EP: np.ndarray,
                        time: np.ndarray,
                        emg_amplitude: float = 0.05,
                        onset_latency: float = 0.02) -> np.ndarray:
        """
        Add tonic EMG activity before EP onset (simulates incomplete relaxation).
        
        Parameters:
        -----------
        EP : np.ndarray
            Original EP signal
        time : np.ndarray
            Time vector (may include negative values for pre-stimulus)
        emg_amplitude : float
            RMS amplitude of baseline EMG in mV
        onset_latency : float
            EP onset time (EMG only before this)
            
        Returns:
        --------
        EP_with_emg : np.ndarray
            EP with baseline EMG
        """
        # Find onset sample using time vector
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate random EMG-like activity (filtered white noise)
        emg = np.random.randn(len(EP)) * emg_amplitude
        
        # Bandpass filter to EMG range (20-500 Hz)
        sos = scipy_signal.butter(4, [20, 500], btype='band', 
                                 fs=self.sampling_rate, output='sos')
        emg = scipy_signal.sosfiltfilt(sos, emg)
        
        # Normalize to desired RMS
        emg = emg / np.std(emg) * emg_amplitude
        
        # Only add before EP onset
        EP_with_emg = EP.copy()
        EP_with_emg[:onset_sample] += emg[:onset_sample]
        
        return EP
    
    def generate_biphasic_EP(self,
                             amplitude: float = 1.0,
                             onset_latency: float = 0.02,
                             phase1_duration: float = 0.010,
                             phase2_duration: float = 0.015,
                             phase2_amplitude_ratio: float = 0.8,
                             total_duration: float = 0.1,
                             pre_stim_duration: float = 0.02,
                             include_onset_dip: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a bi-phasic EP (clear positive-negative pattern).
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude of first phase in mV (default: 1.0 mV)
        onset_latency : float
            Time from stimulus to EP onset in seconds (default: 20 ms)
        phase1_duration : float
            Duration of first phase in seconds (default: 10 ms)
        phase2_duration : float
            Duration of second phase in seconds (default: 15 ms)
        phase2_amplitude_ratio : float
            Ratio of phase 2 to phase 1 amplitude (default: 0.8 for clear negative)
        total_duration : float
            Total signal duration AFTER stimulus in seconds (default: 100 ms)
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms)
        include_onset_dip : bool
            Whether to include physiologically realistic onset ramp with pre-onset dip (default: True)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        EP : np.ndarray
            Bi-phasic EP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize EP signal
        EP = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate time vector for EP
        t_EP = time[onset_sample:] - onset_latency
        
        # Phase 1: Sharp positive deflection using Gaussian-like shape
        t1_peak = phase1_duration * 0.4  # Peak at 40% of phase 1
        sigma1 = phase1_duration / 4
        phase1 = amplitude * np.exp(-((t_EP - t1_peak) ** 2) / (2 * sigma1 ** 2))
        
        # Phase 2: Broader negative deflection
        t2_start = phase1_duration * 0.7
        t2_peak = t2_start + phase2_duration * 0.5
        t_shifted = t_EP - t2_start
        t_shifted[t_shifted < 0] = 0
        
        sigma2 = phase2_duration / 3
        phase2 = -amplitude * phase2_amplitude_ratio * np.exp(-((t_EP - t2_peak) ** 2) / (2 * sigma2 ** 2))
        
        # Combine both phases
        EP_signal = phase1 + phase2
        
        # CRITICAL: Add smooth onset ramp with pre-onset dip (physiologically realistic)
        # Only apply if include_onset_dip is True
        if include_onset_dip:
            # Increased duration and adjusted parameters for more pronounced dip
            onset_ramp_duration = min(0.005, phase1_duration * 0.5)  # 5 ms or 50% of phase 1 (longer)
            onset_ramp = np.zeros_like(t_EP)
            ramp_mask = t_EP <= onset_ramp_duration
            # Adjusted tanh parameters: shallower slope creates more gradual ramp with deeper undershoot
            onset_ramp[ramp_mask] = 0.5 * (1 + np.tanh((t_EP[ramp_mask] - onset_ramp_duration/2) / (onset_ramp_duration/8)))
            onset_ramp[~ramp_mask] = 1.0  # Full amplitude after ramp
            
            EP_signal = EP_signal * onset_ramp
        
        # Apply gentle envelope for realistic decay
        total_EP_duration = phase1_duration + phase2_duration
        envelope = np.exp(-t_EP / (total_EP_duration * 1.5))
        EP_signal = EP_signal * envelope
        
        # Insert into full signal
        EP[onset_sample:onset_sample + len(EP_signal)] = EP_signal
        
        # REVIEWER FIX: Apply smoothing for perfectly smooth waveforms
        from scipy.signal import savgol_filter
        window_length = min(101, max(51, len(EP) // 15))
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 5:
            EP = savgol_filter(EP, window_length, polyorder=3)
        
        return time, EP
    
    def generate_triphasic_EP(self,
                              amplitude: float = 1.0,
                              onset_latency: float = 0.02,
                              phase1_duration: float = 0.008,
                              phase2_duration: float = 0.012,
                              phase3_duration: float = 0.010,
                              phase2_ratio: float = 0.75,
                              phase3_ratio: float = 0.4,
                              total_duration: float = 0.1,
                              pre_stim_duration: float = 0.02,
                              include_onset_dip: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a tri-phasic EP (clear positive-negative-positive pattern).
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude of first phase in mV (default: 1.0 mV)
        onset_latency : float
            Time from stimulus to EP onset in seconds (default: 20 ms)
        phase1_duration : float
            Duration of first positive phase in seconds (default: 8 ms)
        phase2_duration : float
            Duration of negative phase in seconds (default: 12 ms)
        phase3_duration : float
            Duration of final positive phase in seconds (default: 10 ms)
        phase2_ratio : float
            Ratio of phase 2 to phase 1 amplitude (default: 0.75 for clear negative)
        phase3_ratio : float
            Ratio of phase 3 to phase 1 amplitude (default: 0.4 for visible recovery)
        total_duration : float
            Total signal duration AFTER stimulus in seconds (default: 100 ms)
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms)
        include_onset_dip : bool
            Whether to include physiologically realistic onset ramp with pre-onset dip (default: True)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        EP : np.ndarray
            Tri-phasic EP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize EP signal
        EP = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate time vector for EP
        t_EP = time[onset_sample:] - onset_latency
        
        # Phase 1: Initial sharp positive deflection
        t1_peak = phase1_duration * 0.35
        sigma1 = phase1_duration / 5
        phase1 = amplitude * np.exp(-((t_EP - t1_peak) ** 2) / (2 * sigma1 ** 2))
        
        # Phase 2: Clear negative deflection
        t2_start = phase1_duration * 0.65
        t2_peak = t2_start + phase2_duration * 0.45
        sigma2 = phase2_duration / 3.5
        phase2 = -amplitude * phase2_ratio * np.exp(-((t_EP - t2_peak) ** 2) / (2 * sigma2 ** 2))
        
        # Phase 3: Smaller positive recovery deflection  
        t3_start = t2_start + phase2_duration * 0.8
        t3_peak = t3_start + phase3_duration * 0.4
        sigma3 = phase3_duration / 3
        phase3 = amplitude * phase3_ratio * np.exp(-((t_EP - t3_peak) ** 2) / (2 * sigma3 ** 2))
        
        # Combine all three phases
        EP_signal = phase1 + phase2 + phase3
        
        # CRITICAL: Add smooth onset ramp with pre-onset dip (physiologically realistic)
        # Only apply if include_onset_dip is True
        if include_onset_dip:
            # Increased duration and adjusted parameters for more pronounced dip
            onset_ramp_duration = min(0.005, phase1_duration * 0.5)  # 5 ms or 50% of phase 1 (longer)
            onset_ramp = np.zeros_like(t_EP)
            ramp_mask = t_EP <= onset_ramp_duration
            # Adjusted tanh parameters: shallower slope creates more gradual ramp with deeper undershoot
            onset_ramp[ramp_mask] = 0.5 * (1 + np.tanh((t_EP[ramp_mask] - onset_ramp_duration/2) / (onset_ramp_duration/8)))
            onset_ramp[~ramp_mask] = 1.0  # Full amplitude after ramp
            
            EP_signal = EP_signal * onset_ramp
        
        # Apply gentle envelope to ensure return to baseline
        total_EP_duration = phase1_duration + phase2_duration + phase3_duration
        envelope = np.exp(-t_EP / (total_EP_duration * 1.2))
        EP_signal = EP_signal * envelope
        
        # Insert into full signal
        EP[onset_sample:onset_sample + len(EP_signal)] = EP_signal
        
        # REVIEWER FIX: Apply smoothing for perfectly smooth waveforms
        from scipy.signal import savgol_filter
        window_length = min(101, max(51, len(EP) // 15))
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 5:
            EP = savgol_filter(EP, window_length, polyorder=3)
        
        return time, EP

