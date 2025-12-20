"""
Noise Generator
Generate various types of noise commonly found in MEP recordings.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple


class NoiseGenerator:
    """Generate realistic noise for MEP recordings."""
    
    def __init__(self, sampling_rate: int = 2000):
        """
        Initialize noise generator.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz (default: 2000 Hz)
        """
        self.sampling_rate = sampling_rate
        
    def add_white_noise(self, 
                       signal: np.ndarray,
                       snr_db: float = 20.0) -> np.ndarray:
        """
        Add white Gaussian noise to signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        snr_db : float
            Signal-to-noise ratio in dB
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added white noise
        """
        # Calculate signal power (excluding zeros)
        signal_power = np.mean(signal[signal != 0] ** 2)
        
        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate white noise
        noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
        
        return signal + noise
    
    def add_line_noise(self,
                      signal: np.ndarray,
                      time: np.ndarray,
                      frequency: float = 50.0,
                      amplitude: float = 0.1,
                      harmonics: int = 2) -> np.ndarray:
        """
        Add power line noise (50 Hz or 60 Hz with harmonics).
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        time : np.ndarray
            Time vector in seconds
        frequency : float
            Line frequency in Hz (50 for UK/Europe, 60 for US)
        amplitude : float
            Amplitude of line noise in mV
        harmonics : int
            Number of harmonics to include (e.g., 2 = 50, 100, 150 Hz)
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added line noise
        """
        line_noise = np.zeros_like(signal)
        
        for h in range(1, harmonics + 1):
            # Add fundamental and harmonics with decreasing amplitude
            harmonic_amplitude = amplitude / h
            line_noise += harmonic_amplitude * np.sin(2 * np.pi * frequency * h * time)
        
        return signal + line_noise
    
    def add_emg_noise(self,
                     signal: np.ndarray,
                     amplitude: float = 0.05,
                     burst_probability: float = 0.1) -> np.ndarray:
        """
        Add EMG-like noise (simulates tonic muscle activity and occasional bursts).
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        amplitude : float
            RMS amplitude of EMG noise in mV
        burst_probability : float
            Probability of EMG bursts per sample (0-1)
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added EMG noise
        """
        # Generate filtered white noise (20-500 Hz for EMG)
        emg = np.random.randn(len(signal))
        sos = scipy_signal.butter(4, [20, 500], btype='band', 
                                 fs=self.sampling_rate, output='sos')
        emg = scipy_signal.sosfiltfilt(sos, emg)
        
        # Normalize to desired RMS
        emg = emg / np.std(emg) * amplitude
        
        # Add occasional bursts
        if burst_probability > 0:
            burst_mask = np.random.rand(len(signal)) < burst_probability
            # Smooth burst mask
            burst_mask = scipy_signal.convolve(burst_mask.astype(float), 
                                              np.ones(int(self.sampling_rate * 0.05)), 
                                              mode='same')
            burst_mask = burst_mask > 0.5
            emg[burst_mask] *= 3  # 3x amplitude during bursts
        
        return signal + emg
    
    def add_ecg_artifact(self,
                        signal: np.ndarray,
                        time: np.ndarray,
                        heart_rate: float = 70.0,
                        amplitude: float = 0.2) -> np.ndarray:
        """
        Add ECG artifact (simplified QRS complex).
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        time : np.ndarray
            Time vector in seconds
        heart_rate : float
            Heart rate in beats per minute
        amplitude : float
            ECG amplitude in mV
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added ECG artifact
        """
        # Calculate R-R interval
        rr_interval = 60.0 / heart_rate
        
        # Generate QRS complexes at intervals
        ecg = np.zeros_like(signal)
        
        t_qrs = 0
        while t_qrs < time[-1]:
            # Find nearest sample
            idx = np.argmin(np.abs(time - t_qrs))
            
            # Generate simplified QRS complex (biphasic spike)
            qrs_duration = int(0.08 * self.sampling_rate)  # 80 ms QRS
            if idx + qrs_duration < len(signal):
                t_qrs_local = np.linspace(0, 0.08, qrs_duration)
                qrs = amplitude * np.exp(-((t_qrs_local - 0.04) ** 2) / (0.01 ** 2))
                qrs -= 0.3 * amplitude * np.exp(-((t_qrs_local - 0.05) ** 2) / (0.015 ** 2))
                ecg[idx:idx + qrs_duration] += qrs
            
            # Add variability to heart rate
            t_qrs += rr_interval * (1 + 0.1 * np.random.randn())
        
        return signal + ecg
    
    def add_movement_artifact(self,
                             signal: np.ndarray,
                             time: np.ndarray,
                             amplitude: float = 0.3,
                             frequency: float = 0.5) -> np.ndarray:
        """
        Add slow movement artifact (low-frequency drift).
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        time : np.ndarray
            Time vector in seconds
        amplitude : float
            Maximum amplitude of movement in mV
        frequency : float
            Approximate frequency of movement in Hz
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added movement artifact
        """
        # Generate slow sinusoidal drift with some randomness
        movement = amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Add random walk component
        random_walk = np.cumsum(np.random.randn(len(signal))) * amplitude * 0.01
        random_walk = random_walk - np.mean(random_walk)  # Remove DC offset
        
        # Low-pass filter to keep it slow
        sos = scipy_signal.butter(2, 2, btype='low', 
                                 fs=self.sampling_rate, output='sos')
        movement = scipy_signal.sosfiltfilt(sos, movement + random_walk)
        
        return signal + movement
    
    def add_amplifier_noise(self,
                           signal: np.ndarray,
                           amplitude: float = 0.005) -> np.ndarray:
        """
        Add amplifier noise (very low amplitude, broadband).
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        amplitude : float
            RMS amplitude in mV (typically 1-5 µV)
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added amplifier noise
        """
        noise = np.random.randn(len(signal)) * amplitude
        return signal + noise
    
    def add_tms_artifact(self,
                        signal: np.ndarray,
                        time: np.ndarray,
                        stimulus_time: float = 0.0,
                        amplitude: float = 1.5,
                        decay_time: float = 0.002) -> np.ndarray:
        """
        Add TMS stimulus artifact (brief spike with fast exponential decay).
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        time : np.ndarray
            Time vector in seconds
        stimulus_time : float
            Time of TMS pulse in seconds
        amplitude : float
            Peak amplitude of artifact in mV (default: 1.5 mV - realistic for processed signal)
        decay_time : float
            Exponential decay time constant in seconds (default: 2 ms - fast decay)
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with added TMS artifact
        
        Notes:
        ------
        This simulates a post-processed TMS artifact (after blanking/suppression).
        Raw TMS artifacts can be much larger but are typically handled before analysis.
        """
        # Find stimulus sample
        stim_idx = np.argmin(np.abs(time - stimulus_time))
        
        # Generate artifact
        artifact = np.zeros_like(signal)
        t_after_stim = time[stim_idx:] - stimulus_time
        
        # Fast exponential decay with brief oscillation (simulates residual ringing)
        # Use faster decay and lower amplitude oscillation
        decay = amplitude * np.exp(-t_after_stim / decay_time)
        
        # Brief, damped oscillation at lower frequency
        oscillation = np.sin(2 * np.pi * 300 * t_after_stim)  # 300 Hz oscillation
        oscillation_envelope = np.exp(-t_after_stim / (decay_time * 0.3))  # Oscillation dies quickly
        
        # Combine: main decay with small damped oscillation
        artifact[stim_idx:] = decay * (1 + 0.3 * oscillation * oscillation_envelope)
        
        return signal + artifact
    
    def add_composite_noise(self,
                           signal: np.ndarray,
                           time: np.ndarray,
                           snr_db: float = 10.0,
                           include_line: bool = True,
                           include_emg: bool = True,
                           include_ecg: bool = False,
                           include_movement: bool = False,
                           include_tms: bool = False) -> np.ndarray:
        """
        Add composite noise with multiple sources.
        
        Parameters:
        -----------
        signal : np.ndarray
            Original signal
        time : np.ndarray
            Time vector in seconds
        snr_db : float
            Overall signal-to-noise ratio in dB
        include_line : bool
            Include 50 Hz line noise
        include_emg : bool
            Include EMG noise
        include_ecg : bool
            Include ECG artifact
        include_movement : bool
            Include movement artifact
        include_tms : bool
            Include TMS artifact
            
        Returns:
        --------
        noisy_signal : np.ndarray
            Signal with composite noise
        """
        noisy = signal.copy()
        
        if include_tms:
            noisy = self.add_tms_artifact(noisy, time)
        
        if include_line:
            noisy = self.add_line_noise(noisy, time, amplitude=0.05)
        
        if include_emg:
            noisy = self.add_emg_noise(noisy, amplitude=0.03)
        
        if include_ecg:
            noisy = self.add_ecg_artifact(noisy, time, amplitude=0.15)
        
        if include_movement:
            noisy = self.add_movement_artifact(noisy, time, amplitude=0.2)
        
        # Add white noise to achieve desired SNR
        noisy = self.add_white_noise(noisy, snr_db=snr_db)
        
        # Add amplifier noise
        noisy = self.add_amplifier_noise(noisy, amplitude=0.003)
        
        return noisy
