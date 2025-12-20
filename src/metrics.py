"""
Metrics Module
Evaluate filter performance using various metrics.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional


class MEPMetrics:
    """Evaluate MEP filter performance."""
    
    def __init__(self, sampling_rate: int = 2000):
        """
        Initialize metrics module.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def detect_mep_onset(self,
                        signal: np.ndarray,
                        time: np.ndarray,
                        threshold_sd: float = 3.0,
                        baseline_window: Tuple[float, float] = (0.0, 0.015)) -> Optional[float]:
        """
        Detect MEP onset using threshold method.
        
        Parameters:
        -----------
        signal : np.ndarray
            MEP signal
        time : np.ndarray
            Time vector
        threshold_sd : float
            Number of standard deviations above baseline for detection
        baseline_window : tuple
            (start, end) times in seconds for baseline calculation
            
        Returns:
        --------
        onset_time : float or None
            Time of MEP onset in seconds, or None if not detected
        """
        # Calculate baseline statistics
        baseline_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
        baseline = signal[baseline_mask]
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        # Calculate threshold
        threshold = baseline_mean + threshold_sd * baseline_std
        
        # Find first crossing after baseline period
        search_mask = time > baseline_window[1]
        search_signal = signal[search_mask]
        search_time = time[search_mask]
        
        # Rectify signal for onset detection
        rectified = np.abs(search_signal - baseline_mean)
        
        # Find first crossing
        crossings = np.where(rectified > (threshold_sd * baseline_std))[0]
        
        if len(crossings) > 0:
            # Check for sustained crossing (at least 3 consecutive samples)
            for i in range(len(crossings) - 2):
                if crossings[i+1] == crossings[i] + 1 and crossings[i+2] == crossings[i] + 2:
                    return search_time[crossings[i]]
        
        return None
    
    def calculate_peak_amplitude(self,
                                signal: np.ndarray,
                                time: np.ndarray,
                                search_window: Tuple[float, float] = (0.015, 0.080)) -> float:
        """
        Calculate peak-to-peak amplitude.
        
        Parameters:
        -----------
        signal : np.ndarray
            MEP signal
        time : np.ndarray
            Time vector
        search_window : tuple
            (start, end) times in seconds for peak search
            
        Returns:
        --------
        amplitude : float
            Peak-to-peak amplitude in mV
        """
        # Find signal in search window
        window_mask = (time >= search_window[0]) & (time <= search_window[1])
        window_signal = signal[window_mask]
        
        if len(window_signal) == 0:
            return 0.0
        
        # Peak-to-peak
        amplitude = np.max(window_signal) - np.min(window_signal)
        
        return amplitude
    
    def calculate_peak_latency(self,
                              signal: np.ndarray,
                              time: np.ndarray,
                              search_window: Tuple[float, float] = (0.015, 0.080)) -> Optional[float]:
        """
        Calculate time of peak amplitude.
        
        Parameters:
        -----------
        signal : np.ndarray
            MEP signal
        time : np.ndarray
            Time vector
        search_window : tuple
            (start, end) times in seconds for peak search
            
        Returns:
        --------
        peak_time : float or None
            Time of peak in seconds
        """
        window_mask = (time >= search_window[0]) & (time <= search_window[1])
        window_signal = signal[window_mask]
        window_time = time[window_mask]
        
        if len(window_signal) == 0:
            return None
        
        # Find absolute peak
        peak_idx = np.argmax(np.abs(window_signal))
        
        return window_time[peak_idx]
    
    def calculate_area_under_curve(self,
                                   signal: np.ndarray,
                                   time: np.ndarray,
                                   window: Tuple[float, float] = (0.015, 0.080)) -> float:
        """
        Calculate area under curve (integral).
        
        Parameters:
        -----------
        signal : np.ndarray
            MEP signal
        time : np.ndarray
            Time vector
        window : tuple
            (start, end) times in seconds
            
        Returns:
        --------
        auc : float
            Area under curve
        """
        window_mask = (time >= window[0]) & (time <= window[1])
        window_signal = np.abs(signal[window_mask])
        
        # Integrate using trapezoidal rule
        auc = np.trapz(window_signal, dx=1/self.sampling_rate)
        
        return auc
    
    def calculate_snr(self,
                     signal: np.ndarray,
                     time: np.ndarray,
                     signal_window: Tuple[float, float] = (0.015, 0.080),
                     noise_window: Tuple[float, float] = (0.0, 0.015)) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Parameters:
        -----------
        signal : np.ndarray
            MEP signal
        time : np.ndarray
            Time vector
        signal_window : tuple
            Window for signal power calculation
        noise_window : tuple
            Window for noise power calculation
            
        Returns:
        --------
        snr_db : float
            SNR in dB
        """
        # Calculate signal power
        signal_mask = (time >= signal_window[0]) & (time <= signal_window[1])
        signal_power = np.mean(signal[signal_mask] ** 2)
        
        # Calculate noise power
        noise_mask = (time >= noise_window[0]) & (time <= noise_window[1])
        noise_power = np.mean(signal[noise_mask] ** 2)
        
        # Avoid division by zero
        if noise_power < 1e-10:
            noise_power = 1e-10
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    def calculate_correlation(self,
                             signal1: np.ndarray,
                             signal2: np.ndarray) -> float:
        """
        Calculate correlation coefficient between two signals.
        
        Parameters:
        -----------
        signal1 : np.ndarray
            First signal (e.g., ground truth)
        signal2 : np.ndarray
            Second signal (e.g., filtered)
            
        Returns:
        --------
        correlation : float
            Pearson correlation coefficient
        """
        return np.corrcoef(signal1, signal2)[0, 1]
    
    def calculate_rmse(self,
                      signal_true: np.ndarray,
                      signal_pred: np.ndarray,
                      time: np.ndarray,
                      window: Optional[Tuple[float, float]] = None) -> float:
        """
        Calculate root mean square error.
        
        Parameters:
        -----------
        signal_true : np.ndarray
            Ground truth signal
        signal_pred : np.ndarray
            Predicted/filtered signal
        time : np.ndarray
            Time vector
        window : tuple or None
            Optional time window for RMSE calculation
            
        Returns:
        --------
        rmse : float
            Root mean square error
        """
        if window is not None:
            mask = (time >= window[0]) & (time <= window[1])
            true = signal_true[mask]
            pred = signal_pred[mask]
        else:
            true = signal_true
            pred = signal_pred
        
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        
        return rmse
    
    def calculate_all_metrics(self,
                             signal_true: np.ndarray,
                             signal_filtered: np.ndarray,
                             time: np.ndarray,
                             mep_window: Tuple[float, float] = (0.015, 0.080),
                             baseline_window: Tuple[float, float] = (0.0, 0.015)) -> Dict:
        """
        Calculate comprehensive set of metrics.
        
        Parameters:
        -----------
        signal_true : np.ndarray
            Ground truth MEP signal (without noise)
        signal_filtered : np.ndarray
            Filtered noisy signal
        time : np.ndarray
            Time vector
        mep_window : tuple
            Time window for MEP analysis
        baseline_window : tuple
            Time window for baseline analysis
            
        Returns:
        --------
        metrics : dict
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Amplitude metrics
        true_amplitude = self.calculate_peak_amplitude(signal_true, time, mep_window)
        filtered_amplitude = self.calculate_peak_amplitude(signal_filtered, time, mep_window)
        metrics['amplitude_true'] = true_amplitude
        metrics['amplitude_filtered'] = filtered_amplitude
        metrics['amplitude_error_abs'] = filtered_amplitude - true_amplitude
        metrics['amplitude_error_pct'] = (filtered_amplitude - true_amplitude) / true_amplitude * 100 if true_amplitude > 0 else 0
        
        # Timing metrics
        true_onset = self.detect_mep_onset(signal_true, time, baseline_window=baseline_window)
        filtered_onset = self.detect_mep_onset(signal_filtered, time, baseline_window=baseline_window)
        metrics['onset_true'] = true_onset
        metrics['onset_filtered'] = filtered_onset
        if true_onset is not None and filtered_onset is not None:
            metrics['onset_error_ms'] = (filtered_onset - true_onset) * 1000
        else:
            metrics['onset_error_ms'] = np.nan
        
        true_peak_time = self.calculate_peak_latency(signal_true, time, mep_window)
        filtered_peak_time = self.calculate_peak_latency(signal_filtered, time, mep_window)
        metrics['peak_latency_true'] = true_peak_time
        metrics['peak_latency_filtered'] = filtered_peak_time
        if true_peak_time is not None and filtered_peak_time is not None:
            metrics['peak_latency_error_ms'] = (filtered_peak_time - true_peak_time) * 1000
        else:
            metrics['peak_latency_error_ms'] = np.nan
        
        # Area under curve
        true_auc = self.calculate_area_under_curve(signal_true, time, mep_window)
        filtered_auc = self.calculate_area_under_curve(signal_filtered, time, mep_window)
        metrics['auc_true'] = true_auc
        metrics['auc_filtered'] = filtered_auc
        metrics['auc_error_pct'] = (filtered_auc - true_auc) / true_auc * 100 if true_auc > 0 else 0
        
        # Morphology metrics
        metrics['correlation'] = self.calculate_correlation(signal_true, signal_filtered)
        metrics['rmse_full'] = self.calculate_rmse(signal_true, signal_filtered, time)
        metrics['rmse_mep'] = self.calculate_rmse(signal_true, signal_filtered, time, mep_window)
        
        # SNR improvement
        metrics['snr_filtered'] = self.calculate_snr(signal_filtered, time, mep_window, baseline_window)
        
        # Baseline stability (variance in baseline period)
        baseline_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
        metrics['baseline_std'] = np.std(signal_filtered[baseline_mask])
        
        return metrics
    
    def detect_spurious_oscillations(self,
                                    signal: np.ndarray,
                                    time: np.ndarray,
                                    baseline_window: Tuple[float, float] = (0.0, 0.015),
                                    threshold_factor: float = 5.0) -> bool:
        """
        Detect spurious oscillations (ringing artifacts) in baseline.
        
        Parameters:
        -----------
        signal : np.ndarray
            Filtered signal
        time : np.ndarray
            Time vector
        baseline_window : tuple
            Baseline time window
        threshold_factor : float
            Factor above baseline noise to consider oscillation
            
        Returns:
        --------
        has_oscillations : bool
            True if spurious oscillations detected
        """
        baseline_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
        baseline = signal[baseline_mask]
        
        # Calculate zero crossings in baseline
        zero_crossings = np.sum(np.diff(np.sign(baseline)) != 0)
        
        # Calculate baseline envelope
        baseline_envelope = np.abs(scipy_signal.hilbert(baseline - np.mean(baseline)))
        
        # Check if envelope exceeds threshold
        baseline_std = np.std(baseline)
        excessive_oscillations = np.any(baseline_envelope > threshold_factor * baseline_std)
        
        # Also check crossing rate
        crossing_rate = zero_crossings / len(baseline)
        high_crossing_rate = crossing_rate > 0.3  # More than 30% of samples
        
        return excessive_oscillations or high_crossing_rate
