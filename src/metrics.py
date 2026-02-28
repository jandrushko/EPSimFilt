"""
Metrics Module
Evaluate filter performance using various metrics.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Tuple, Optional


class EPMetrics:
    """Evaluate EP filter performance."""
    
    def __init__(self, sampling_rate: int = 2000):
        """
        Initialize metrics module.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
    def detect_EP_onset(self,
                        signal: np.ndarray,
                        time: np.ndarray,
                        threshold_sd: float = 2.0,
                        baseline_window: Tuple[float, float] = (0.0, 0.015)) -> Optional[float]:
        """
        Detect EP onset using robust peak-first backwards search method.
        
        This method is much more reliable than threshold-based detection because:
        1. First finds the main EP peak (most prominent feature)
        2. Works backwards from peak to find where signal starts rising
        3. Less sensitive to noise and artifacts
        
        Parameters:
        -----------
        signal : np.ndarray
            EP signal
        time : np.ndarray
            Time vector
        threshold_sd : float
            Number of standard deviations above baseline for onset detection (default: 2.0)
        baseline_window : tuple
            (start, end) times in seconds for baseline calculation
            
        Returns:
        --------
        onset_time : float or None
            Time of EP onset in seconds, or None if not detected
            
        Notes:
        ------
        Algorithm:
        1. Calculate baseline mean and std
        2. Find the largest peak (or trough) in expected EP window
        3. Walk backwards from peak until signal returns to near baseline
        4. Onset is where signal first deviates from baseline
        """
        # Calculate baseline statistics
        baseline_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
        baseline = signal[baseline_mask]
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        # Define search window for EP (typically 15-80 ms after stimulus)
        ep_window_start = 0.015  # 15 ms
        ep_window_end = 0.080    # 80 ms
        ep_mask = (time >= ep_window_start) & (time <= ep_window_end)
        
        if not np.any(ep_mask):
            return None
        
        ep_signal = signal[ep_mask]
        ep_time = time[ep_mask]
        
        # Find the main peak (could be positive or negative)
        peak_idx_pos = np.argmax(ep_signal)
        peak_idx_neg = np.argmin(ep_signal)
        
        peak_val_pos = ep_signal[peak_idx_pos]
        peak_val_neg = ep_signal[peak_idx_neg]
        
        # Determine which peak is more prominent (further from baseline)
        prominence_pos = abs(peak_val_pos - baseline_mean) / baseline_std
        prominence_neg = abs(peak_val_neg - baseline_mean) / baseline_std
        
        if prominence_pos > prominence_neg:
            peak_idx = peak_idx_pos
            peak_polarity = 'positive'
        else:
            peak_idx = peak_idx_neg
            peak_polarity = 'negative'
        
        # Now walk backwards from the peak to find onset
        # Onset is where the signal first rises above (or falls below) baseline
        onset_threshold = baseline_mean + threshold_sd * baseline_std * (1 if peak_polarity == 'positive' else -1)
        
        # Start from peak and go backwards
        for i in range(peak_idx, -1, -1):
            # Check if we've returned to baseline
            if peak_polarity == 'positive':
                if ep_signal[i] < onset_threshold:
                    # Found where signal drops below threshold
                    # Onset is the next sample forward
                    if i < len(ep_signal) - 1:
                        return ep_time[i + 1]
                    else:
                        return ep_time[i]
            else:  # negative peak
                if ep_signal[i] > onset_threshold:
                    # Found where signal rises above threshold
                    if i < len(ep_signal) - 1:
                        return ep_time[i + 1]
                    else:
                        return ep_time[i]
        
        # If we walked all the way back to the start without finding onset,
        # use the start of the EP window
        return ep_time[0]
    
    def calculate_peak_amplitude(self,
                                signal: np.ndarray,
                                time: np.ndarray,
                                search_window: Tuple[float, float] = (0.015, 0.080)) -> float:
        """
        Calculate peak-to-peak amplitude.
        
        Parameters:
        -----------
        signal : np.ndarray
            EP signal
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
            EP signal
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
            EP signal
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
        auc = np.trapezoid(window_signal, dx=1/self.sampling_rate)
        
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
            EP signal
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
    
    def fishers_z_transform(self, r: float) -> float:
        """
        Apply Fisher's r-to-z transformation to correlation coefficient.
        
        CRITICAL: Correlation coefficients must be z-transformed before statistical testing
        because the sampling distribution of r is not normal, especially for values far from 0.
        
        Fisher's z = 0.5 * ln((1 + r) / (1 - r)) = arctanh(r)
        
        Parameters:
        -----------
        r : float
            Pearson correlation coefficient (-1 to +1)
            
        Returns:
        --------
        z : float
            Fisher's z-transformed value
            
        Notes:
        ------
        - The z-distribution is approximately normal with variance 1/(n-3)
        - This transformation is essential for:
          1. Comparing correlations between groups
          2. Testing if a correlation differs from a specific value
          3. Averaging correlation coefficients
        - REVIEWER REQUIREMENT: Always report both r and z in outputs
        
        References:
        -----------
        Fisher, R. A. (1915). Frequency distribution of the values of the correlation 
        coefficient in samples from an indefinitely large population. Biometrika, 10(4), 507-521.
        """
        # Handle edge cases
        if r >= 1.0:
            r = 0.9999  # Prevent inf
        elif r <= -1.0:
            r = -0.9999  # Prevent -inf
        
        # Fisher's z transformation: z = arctanh(r)
        z = np.arctanh(r)
        
        return z
    
    def inverse_fishers_z(self, z: float) -> float:
        """
        Convert Fisher's z back to correlation coefficient.
        
        r = tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)
        
        Parameters:
        -----------
        z : float
            Fisher's z-transformed value
            
        Returns:
        --------
        r : float
            Pearson correlation coefficient
        """
        return np.tanh(z)

    
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
                             EP_window: Tuple[float, float] = (0.015, 0.080),
                             baseline_window: Tuple[float, float] = (0.0, 0.015)) -> Dict:
        """
        Calculate comprehensive set of metrics.
        
        Parameters:
        -----------
        signal_true : np.ndarray
            Ground truth EP signal (without noise)
        signal_filtered : np.ndarray
            Filtered noisy signal
        time : np.ndarray
            Time vector
        EP_window : tuple
            Time window for EP analysis
        baseline_window : tuple
            Time window for baseline analysis
            
        Returns:
        --------
        metrics : dict
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Amplitude metrics
        true_amplitude = self.calculate_peak_amplitude(signal_true, time, EP_window)
        filtered_amplitude = self.calculate_peak_amplitude(signal_filtered, time, EP_window)
        metrics['amplitude_true'] = true_amplitude
        metrics['amplitude_filtered'] = filtered_amplitude
        metrics['amplitude_error_abs'] = filtered_amplitude - true_amplitude
        metrics['amplitude_error_pct'] = (filtered_amplitude - true_amplitude) / true_amplitude * 100 if true_amplitude > 0 else 0
        
        # Timing metrics
        true_onset = self.detect_EP_onset(signal_true, time, baseline_window=baseline_window)
        filtered_onset = self.detect_EP_onset(signal_filtered, time, baseline_window=baseline_window)
        metrics['onset_true'] = true_onset
        metrics['onset_filtered'] = filtered_onset
        if true_onset is not None and filtered_onset is not None:
            metrics['onset_error_ms'] = (filtered_onset - true_onset) * 1000
        else:
            metrics['onset_error_ms'] = np.nan
        
        true_peak_time = self.calculate_peak_latency(signal_true, time, EP_window)
        filtered_peak_time = self.calculate_peak_latency(signal_filtered, time, EP_window)
        metrics['peak_latency_true'] = true_peak_time
        metrics['peak_latency_filtered'] = filtered_peak_time
        if true_peak_time is not None and filtered_peak_time is not None:
            metrics['peak_latency_error_ms'] = (filtered_peak_time - true_peak_time) * 1000
        else:
            metrics['peak_latency_error_ms'] = np.nan
        
        # Area under curve
        true_auc = self.calculate_area_under_curve(signal_true, time, EP_window)
        filtered_auc = self.calculate_area_under_curve(signal_filtered, time, EP_window)
        metrics['auc_true'] = true_auc
        metrics['auc_filtered'] = filtered_auc
        metrics['auc_error_pct'] = (filtered_auc - true_auc) / true_auc * 100 if true_auc > 0 else 0
        
        # Morphology metrics
        metrics['correlation'] = self.calculate_correlation(signal_true, signal_filtered)
        # REVIEWER REQUIREMENT: Add Fisher's z-transformed correlation for valid statistics
        metrics['correlation_z'] = self.fishers_z_transform(metrics['correlation'])
        metrics['rmse_full'] = self.calculate_rmse(signal_true, signal_filtered, time)
        metrics['rmse_EP'] = self.calculate_rmse(signal_true, signal_filtered, time, EP_window)
        
        # SNR improvement
        metrics['snr_filtered'] = self.calculate_snr(signal_filtered, time, EP_window, baseline_window)
        
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
    
    def apply_log_transform(self, data: np.ndarray, shift: float = 1e-10) -> np.ndarray:
        """
        Apply log transformation to data (useful for normalizing skewed distributions).
        
        Parameters:
        -----------
        data : np.ndarray
            Data to transform (can contain negative values)
        shift : float
            Small constant to ensure positivity (default: 1e-10)
            
        Returns:
        --------
        log_data : np.ndarray
            Log-transformed data
            
        Notes:
        ------
        For data that can be negative (like amplitude error), we use:
        log_transform = sign(x) * log(abs(x) + shift)
        This preserves the sign while allowing log transformation.
        """
        # Handle negative values by preserving sign
        sign = np.sign(data)
        abs_data = np.abs(data)
        
        # Apply log transformation with small shift to avoid log(0)
        log_data = sign * np.log(abs_data + shift)
        
        return log_data
    
    def test_bimodality_coefficient(self, data: np.ndarray) -> Dict:
        """
        Test for bimodality using bimodality coefficient.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to test for bimodality
            
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'bc': Bimodality coefficient
            - 'skewness': Skewness of distribution
            - 'kurtosis': Kurtosis of distribution
            - 'is_bimodal': Boolean (True if BC > 0.555)
            
        Notes:
        ------
        Bimodality coefficient (BC) = (skewness^2 + 1) / kurtosis
        BC > 5/9 (≈0.555) suggests bimodality
        This is a simpler alternative to Hartigan's dip test
        """
        from scipy import stats
        
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 4:
            return {
                'bc': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'is_bimodal': False
            }
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(data_clean)
        kurtosis_val = stats.kurtosis(data_clean, fisher=False)  # Pearson's kurtosis
        
        # Bimodality coefficient
        if kurtosis_val == 0:
            bc = np.nan
        else:
            bc = (skewness**2 + 1) / kurtosis_val
        
        # Threshold for bimodality
        threshold = 5/9  # ≈0.555
        
        return {
            'bc': bc,
            'skewness': skewness,
            'kurtosis': kurtosis_val,
            'is_bimodal': bc > threshold if not np.isnan(bc) else False
        }
    
    def test_normality(self, data: np.ndarray) -> Dict:
        """
        Test if data follows a normal distribution using multiple tests.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to test for normality
            
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'shapiro_p': Shapiro-Wilk test p-value
            - 'anderson_stat': Anderson-Darling statistic
            - 'is_normal': Boolean (True if normal at α=0.05)
            - 'skewness': Skewness value
            - 'kurtosis': Excess kurtosis
            - 'recommendation': Text recommendation
            
        Notes:
        ------
        Normality tests:
        - Shapiro-Wilk: Most powerful for n < 50, good for n < 2000
        - Anderson-Darling: Good for larger samples
        
        Interpretation:
        - p > 0.05: Data is normally distributed (can use parametric tests)
        - p < 0.05: Data is NOT normal (consider transformation or non-parametric tests)
        - Skewness ≈ 0 and Kurtosis ≈ 0 indicate normality
        """
        from scipy import stats
        
        # Remove NaN values
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) < 3:
            return {
                'shapiro_p': np.nan,
                'anderson_stat': np.nan,
                'is_normal': False,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'recommendation': 'Insufficient data'
            }
        
        # Shapiro-Wilk test (works for n=3 to 5000)
        if len(data_clean) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data_clean)
        else:
            shapiro_p = np.nan
        
        # Anderson-Darling test
        anderson_result = stats.anderson(data_clean, dist='norm')
        anderson_stat = anderson_result.statistic
        
        # Skewness and kurtosis
        skewness = stats.skew(data_clean)
        kurtosis = stats.kurtosis(data_clean)  # Excess kurtosis (0 = normal)
        
        # Determine if normal
        is_normal = shapiro_p > 0.05 if not np.isnan(shapiro_p) else anderson_stat < anderson_result.critical_values[2]
        
        # Generate recommendation
        if is_normal:
            recommendation = "✓ Data is normally distributed - parametric tests appropriate"
        else:
            if abs(skewness) > 1:
                recommendation = "⚠️ Highly skewed - log transformation recommended"
            elif abs(skewness) > 0.5:
                recommendation = "⚠️ Moderately skewed - consider log transformation"
            else:
                recommendation = "⚠️ Non-normal but not skewed - check for outliers or bimodality"
        
        return {
            'shapiro_p': shapiro_p,
            'anderson_stat': anderson_stat,
            'is_normal': is_normal,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'recommendation': recommendation
        }
    
    def test_homogeneity_of_variance(self, groups: list) -> Dict:
        """
        Test if multiple groups have equal variances (homoscedasticity).
        
        Parameters:
        -----------
        groups : list of np.ndarray
            List of data arrays for different groups/conditions
            
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'levene_p': Levene's test p-value
            - 'bartlett_p': Bartlett's test p-value
            - 'is_homogeneous': Boolean (True if equal variances)
            - 'recommendation': Text recommendation
            
        Notes:
        ------
        - Levene's test: Robust to non-normality
        - Bartlett's test: More powerful but assumes normality
        
        Interpretation:
        - p > 0.05: Variances are equal (can use ANOVA, t-test)
        - p < 0.05: Variances are unequal (use Welch's test or transformation)
        """
        from scipy import stats
        
        # Remove NaN values from each group
        groups_clean = [g[~np.isnan(g)] for g in groups]
        groups_clean = [g for g in groups_clean if len(g) >= 2]
        
        if len(groups_clean) < 2:
            return {
                'levene_p': np.nan,
                'bartlett_p': np.nan,
                'is_homogeneous': False,
                'recommendation': 'Insufficient groups for comparison'
            }
        
        # Levene's test (robust to non-normality)
        levene_stat, levene_p = stats.levene(*groups_clean)
        
        # Bartlett's test (assumes normality)
        try:
            bartlett_stat, bartlett_p = stats.bartlett(*groups_clean)
        except:
            bartlett_p = np.nan
        
        # Determine if homogeneous
        is_homogeneous = levene_p > 0.05
        
        # Generate recommendation
        if is_homogeneous:
            recommendation = "✓ Equal variances - standard ANOVA/t-test appropriate"
        else:
            recommendation = "⚠️ Unequal variances - use Welch's test or log transformation"
        
        return {
            'levene_p': levene_p,
            'bartlett_p': bartlett_p,
            'is_homogeneous': is_homogeneous,
            'recommendation': recommendation
        }

