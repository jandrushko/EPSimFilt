"""
MEP Signal Generator
Generate realistic Motor Evoked Potential (MEP) waveforms with configurable parameters.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal as scipy_signal


class MEPGenerator:
    """Generate realistic MEP waveforms."""
    
    def __init__(self, sampling_rate: int = 2000):
        """
        Initialize MEP generator.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz (default: 2000 Hz)
        """
        self.sampling_rate = sampling_rate
        
    def generate_mep(self, 
                     amplitude: float = 1.0,
                     duration: float = 0.03,
                     onset_latency: float = 0.02,
                     rise_time: float = 0.008,
                     decay_time: float = 0.015,
                     total_duration: float = 0.1,
                     asymmetry: float = 1.2,
                     pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a realistic MEP waveform.
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude in mV (default: 1.0 mV)
        duration : float
            Total MEP duration in seconds (default: 30 ms)
        onset_latency : float
            Time from stimulus to MEP onset in seconds (default: 20 ms)
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
        mep : np.ndarray
            MEP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize MEP signal
        mep = np.zeros(n_samples)
        
        # Find onset sample (at specified latency after t=0 stimulus)
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate biphasic waveform using gamma distributions
        # This creates a realistic MEP shape with asymmetric rise and fall
        
        # Calculate parameters for realistic shape
        t_mep = time[onset_sample:] - onset_latency
        
        # Positive phase (main deflection)
        peak_time = rise_time
        k_rise = 3.0  # Shape parameter
        theta_rise = rise_time / k_rise
        
        positive_phase = (t_mep / theta_rise) ** k_rise * np.exp(-k_rise * t_mep / rise_time)
        positive_phase = positive_phase / np.max(positive_phase) * amplitude
        
        # Negative phase (smaller, slower)
        k_decay = 2.0
        theta_decay = decay_time / k_decay
        t_shifted = t_mep - rise_time
        t_shifted[t_shifted < 0] = 0
        
        negative_phase = -(t_shifted / theta_decay) ** k_decay * np.exp(-k_decay * t_shifted / (decay_time * asymmetry))
        negative_phase = negative_phase / np.abs(np.min(negative_phase)) * amplitude * 0.3  # 30% of positive
        
        # Combine phases
        mep_signal = positive_phase + negative_phase
        
        # Apply envelope to ensure it returns to baseline
        envelope = np.exp(-t_mep / (duration * 0.8))
        mep_signal = mep_signal * envelope
        
        # Insert into full signal
        mep[onset_sample:onset_sample + len(mep_signal)] = mep_signal
        
        return time, mep
    
    def generate_template_based_mep(self,
                                   template: np.ndarray,
                                   amplitude: float = 1.0,
                                   onset_latency: float = 0.02,
                                   total_duration: float = 0.1,
                                   pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate MEP based on a real MEP template.
        
        Parameters:
        -----------
        template : np.ndarray
            Template MEP waveform (will be normalized)
        amplitude : float
            Desired amplitude in mV
        onset_latency : float
            Time from stimulus to MEP onset in seconds
        total_duration : float
            Total signal duration AFTER stimulus in seconds
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        mep : np.ndarray
            MEP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Normalize template
        template_normalized = template / np.max(np.abs(template)) * amplitude
        
        # Initialize MEP signal
        mep = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Insert template
        template_len = len(template_normalized)
        if onset_sample + template_len <= n_samples:
            mep[onset_sample:onset_sample + template_len] = template_normalized
        else:
            mep[onset_sample:] = template_normalized[:n_samples - onset_sample]
            
        return time, mep
    
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
        Generate bi-phasic MEP with independent control over each phase.
        
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
            Time from stimulus to MEP onset in seconds
        total_duration : float
            Total signal duration AFTER stimulus in seconds
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative = pre-stimulus)
        mep : np.ndarray
            Bi-phasic MEP waveform in mV
        """
        # Create time vector
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize signal
        mep = np.zeros(n_samples)
        
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
        mep[onset_sample:end_idx1] = phase1[:end_idx1 - onset_sample]
        
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
        mep[phase2_onset_sample:end_idx2] += phase2[:end_idx2 - phase2_onset_sample]
        
        return time, mep
    
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
        Generate tri-phasic MEP with independent control over each phase.
        
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
            Time from stimulus to MEP onset (seconds)
        total_duration : float
            Total signal duration AFTER stimulus (seconds)
        pre_stim_duration : float
            Pre-stimulus baseline duration (seconds)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds
        mep : np.ndarray
            Tri-phasic MEP waveform in mV
        """
        # Create time vector
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize signal
        mep = np.zeros(n_samples)
        
        # Helper function to create Gaussian phase
        def create_phase(amplitude, rise_time, decay_time, polarity=1):
            duration = rise_time + decay_time
            n_samp = int(duration * self.sampling_rate)
            t = np.linspace(0, duration, n_samp)
            peak_time = rise_time
            sigma = rise_time / 2.5
            phase = polarity * amplitude * np.exp(-((t - peak_time)**2) / (2 * sigma**2))
            
            # Smooth onset
            ramp_dur = min(0.003, rise_time * 0.3)
            ramp_samp = int(ramp_dur * self.sampling_rate)
            if ramp_samp > 0 and ramp_samp < len(phase):
                ramp = 0.5 * (1 + np.tanh((np.linspace(0, 3, ramp_samp) - 1.5) / 0.5))
                phase[:ramp_samp] *= ramp
            
            return phase
        
        # PHASE 1: Positive
        onset_sample = np.argmin(np.abs(time - onset_latency))
        phase1 = create_phase(phase1_amplitude, phase1_rise_time, phase1_decay_time, polarity=1)
        end_idx1 = min(onset_sample + len(phase1), n_samples)
        mep[onset_sample:end_idx1] = phase1[:end_idx1 - onset_sample]
        
        # PHASE 2: Negative
        phase2_onset = onset_latency + phase1_rise_time + phase1_2_separation
        phase2_onset_sample = np.argmin(np.abs(time - phase2_onset))
        phase2_amp = phase1_amplitude * phase2_amplitude_ratio
        phase2 = create_phase(phase2_amp, phase2_rise_time, phase2_decay_time, polarity=-1)
        end_idx2 = min(phase2_onset_sample + len(phase2), n_samples)
        mep[phase2_onset_sample:end_idx2] += phase2[:end_idx2 - phase2_onset_sample]
        
        # PHASE 3: Positive
        phase3_onset = phase2_onset + phase2_rise_time + phase2_3_separation
        phase3_onset_sample = np.argmin(np.abs(time - phase3_onset))
        phase3_amp = phase1_amplitude * phase3_amplitude_ratio
        phase3 = create_phase(phase3_amp, phase3_rise_time, phase3_decay_time, polarity=1)
        end_idx3 = min(phase3_onset_sample + len(phase3), n_samples)
        mep[phase3_onset_sample:end_idx3] += phase3[:end_idx3 - phase3_onset_sample]
        
        return time, mep
    
    def generate_double_peak_mep(self,
                                amplitude1: float = 1.0,
                                amplitude2: float = 0.6,
                                peak_separation: float = 0.005,
                                onset_latency: float = 0.02,
                                total_duration: float = 0.1,
                                pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate MEP with double peak (occasionally seen with dual coil stimulation).
        
        Parameters:
        -----------
        amplitude1 : float
            First peak amplitude in mV
        amplitude2 : float
            Second peak amplitude in mV
        peak_separation : float
            Time between peaks in seconds (default: 5 ms)
        onset_latency : float
            Time from stimulus to first MEP onset in seconds
        total_duration : float
            Total signal duration AFTER stimulus in seconds
        pre_stim_duration : float
            Duration of pre-stimulus baseline in seconds (default: 20 ms)
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        mep : np.ndarray
            MEP waveform in mV
        """
        # Generate first MEP
        time, mep1 = self.generate_mep(amplitude=amplitude1, 
                                       onset_latency=onset_latency,
                                       total_duration=total_duration,
                                       pre_stim_duration=pre_stim_duration)
        
        # Generate second MEP with offset
        _, mep2 = self.generate_mep(amplitude=amplitude2,
                                    onset_latency=onset_latency + peak_separation,
                                    total_duration=total_duration,
                                    pre_stim_duration=pre_stim_duration)
        
        # Combine
        mep = mep1 + mep2
        
        return time, mep
    
    def add_baseline_emg(self,
                        mep: np.ndarray,
                        time: np.ndarray,
                        emg_amplitude: float = 0.05,
                        onset_latency: float = 0.02) -> np.ndarray:
        """
        Add tonic EMG activity before MEP onset (simulates incomplete relaxation).
        
        Parameters:
        -----------
        mep : np.ndarray
            Original MEP signal
        time : np.ndarray
            Time vector (may include negative values for pre-stimulus)
        emg_amplitude : float
            RMS amplitude of baseline EMG in mV
        onset_latency : float
            MEP onset time (EMG only before this)
            
        Returns:
        --------
        mep_with_emg : np.ndarray
            MEP with baseline EMG
        """
        # Find onset sample using time vector
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate random EMG-like activity (filtered white noise)
        emg = np.random.randn(len(mep)) * emg_amplitude
        
        # Bandpass filter to EMG range (20-500 Hz)
        sos = scipy_signal.butter(4, [20, 500], btype='band', 
                                 fs=self.sampling_rate, output='sos')
        emg = scipy_signal.sosfiltfilt(sos, emg)
        
        # Normalize to desired RMS
        emg = emg / np.std(emg) * emg_amplitude
        
        # Only add before MEP onset
        mep_with_emg = mep.copy()
        mep_with_emg[:onset_sample] += emg[:onset_sample]
        
        return mep
    
    def generate_biphasic_mep(self,
                             amplitude: float = 1.0,
                             onset_latency: float = 0.02,
                             phase1_duration: float = 0.010,
                             phase2_duration: float = 0.015,
                             phase2_amplitude_ratio: float = 0.8,
                             total_duration: float = 0.1,
                             pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a bi-phasic MEP (clear positive-negative pattern).
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude of first phase in mV (default: 1.0 mV)
        onset_latency : float
            Time from stimulus to MEP onset in seconds (default: 20 ms)
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
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        mep : np.ndarray
            Bi-phasic MEP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize MEP signal
        mep = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate time vector for MEP
        t_mep = time[onset_sample:] - onset_latency
        
        # Phase 1: Sharp positive deflection using Gaussian-like shape
        t1_peak = phase1_duration * 0.4  # Peak at 40% of phase 1
        sigma1 = phase1_duration / 4
        phase1 = amplitude * np.exp(-((t_mep - t1_peak) ** 2) / (2 * sigma1 ** 2))
        
        # Phase 2: Broader negative deflection
        t2_start = phase1_duration * 0.7
        t2_peak = t2_start + phase2_duration * 0.5
        t_shifted = t_mep - t2_start
        t_shifted[t_shifted < 0] = 0
        
        sigma2 = phase2_duration / 3
        phase2 = -amplitude * phase2_amplitude_ratio * np.exp(-((t_mep - t2_peak) ** 2) / (2 * sigma2 ** 2))
        
        # Combine both phases
        mep_signal = phase1 + phase2
        
        # CRITICAL: Add smooth onset ramp to prevent vertical jump
        # Use sigmoid/tanh-based smooth onset over first 2-3 ms
        onset_ramp_duration = min(0.003, phase1_duration * 0.3)  # 3 ms or 30% of phase 1
        onset_ramp = np.zeros_like(t_mep)
        ramp_mask = t_mep <= onset_ramp_duration
        onset_ramp[ramp_mask] = 0.5 * (1 + np.tanh((t_mep[ramp_mask] - onset_ramp_duration/2) / (onset_ramp_duration/6)))
        onset_ramp[~ramp_mask] = 1.0  # Full amplitude after ramp
        
        mep_signal = mep_signal * onset_ramp
        
        # Apply gentle envelope for realistic decay
        total_mep_duration = phase1_duration + phase2_duration
        envelope = np.exp(-t_mep / (total_mep_duration * 1.5))
        mep_signal = mep_signal * envelope
        
        # Insert into full signal
        mep[onset_sample:onset_sample + len(mep_signal)] = mep_signal
        
        return time, mep
    
    def generate_triphasic_mep(self,
                              amplitude: float = 1.0,
                              onset_latency: float = 0.02,
                              phase1_duration: float = 0.008,
                              phase2_duration: float = 0.012,
                              phase3_duration: float = 0.010,
                              phase2_ratio: float = 0.75,
                              phase3_ratio: float = 0.4,
                              total_duration: float = 0.1,
                              pre_stim_duration: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a tri-phasic MEP (clear positive-negative-positive pattern).
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude of first phase in mV (default: 1.0 mV)
        onset_latency : float
            Time from stimulus to MEP onset in seconds (default: 20 ms)
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
            
        Returns:
        --------
        time : np.ndarray
            Time vector in seconds (negative values = pre-stimulus)
        mep : np.ndarray
            Tri-phasic MEP waveform in mV
        """
        # Create time vector with pre-stimulus window
        total_time = pre_stim_duration + total_duration
        n_samples = int(total_time * self.sampling_rate)
        time = np.linspace(-pre_stim_duration, total_duration, n_samples)
        
        # Initialize MEP signal
        mep = np.zeros(n_samples)
        
        # Find onset sample
        onset_sample = np.argmin(np.abs(time - onset_latency))
        
        # Generate time vector for MEP
        t_mep = time[onset_sample:] - onset_latency
        
        # Phase 1: Initial sharp positive deflection
        t1_peak = phase1_duration * 0.35
        sigma1 = phase1_duration / 5
        phase1 = amplitude * np.exp(-((t_mep - t1_peak) ** 2) / (2 * sigma1 ** 2))
        
        # Phase 2: Clear negative deflection
        t2_start = phase1_duration * 0.65
        t2_peak = t2_start + phase2_duration * 0.45
        sigma2 = phase2_duration / 3.5
        phase2 = -amplitude * phase2_ratio * np.exp(-((t_mep - t2_peak) ** 2) / (2 * sigma2 ** 2))
        
        # Phase 3: Smaller positive recovery deflection  
        t3_start = t2_start + phase2_duration * 0.8
        t3_peak = t3_start + phase3_duration * 0.4
        sigma3 = phase3_duration / 3
        phase3 = amplitude * phase3_ratio * np.exp(-((t_mep - t3_peak) ** 2) / (2 * sigma3 ** 2))
        
        # Combine all three phases
        mep_signal = phase1 + phase2 + phase3
        
        # CRITICAL: Add smooth onset ramp to prevent vertical jump
        # Use sigmoid/tanh-based smooth onset over first 2-3 ms
        onset_ramp_duration = min(0.003, phase1_duration * 0.3)  # 3 ms or 30% of phase 1
        onset_ramp = np.zeros_like(t_mep)
        ramp_mask = t_mep <= onset_ramp_duration
        onset_ramp[ramp_mask] = 0.5 * (1 + np.tanh((t_mep[ramp_mask] - onset_ramp_duration/2) / (onset_ramp_duration/6)))
        onset_ramp[~ramp_mask] = 1.0  # Full amplitude after ramp
        
        mep_signal = mep_signal * onset_ramp
        
        # Apply gentle envelope to ensure return to baseline
        total_mep_duration = phase1_duration + phase2_duration + phase3_duration
        envelope = np.exp(-t_mep / (total_mep_duration * 1.2))
        mep_signal = mep_signal * envelope
        
        # Insert into full signal
        mep[onset_sample:onset_sample + len(mep_signal)] = mep_signal
        
        return time, mep

