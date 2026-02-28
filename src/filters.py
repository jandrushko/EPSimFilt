"""
Filter Module
Implement various digital filters for EP analysis.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional, Dict

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class EPFilters:
    """Collection of filters for EP signal processing."""
    
    def __init__(self, sampling_rate: int = 2000):
        """
        Initialize filter module.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz (default: 2000 Hz)
        """
        self.sampling_rate = sampling_rate
        
    def butterworth_filter(self,
                          signal: np.ndarray,
                          lowcut: Optional[float] = None,
                          highcut: Optional[float] = None,
                          order: int = 4,
                          zero_phase: bool = True) -> np.ndarray:
        """
        Apply Butterworth filter (bandpass, lowpass, or highpass).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        lowcut : float or None
            High-pass cutoff frequency in Hz (None for no high-pass)
        highcut : float or None
            Low-pass cutoff frequency in Hz (None for no low-pass)
        order : int
            Filter order
        zero_phase : bool
            If True, apply filter forwards and backwards (filtfilt)
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        # Validate cutoff frequencies (must be < Nyquist frequency)
        nyquist = self.sampling_rate / 2.0
        
        if lowcut is not None and lowcut >= nyquist:
            lowcut = nyquist * 0.99  # Set to 99% of Nyquist
        if highcut is not None and highcut >= nyquist:
            highcut = nyquist * 0.99  # Set to 99% of Nyquist
        
        # Ensure lowcut < highcut for bandpass
        if lowcut is not None and highcut is not None and lowcut >= highcut:
            # Swap them or adjust
            lowcut = highcut * 0.5
        
        # Determine filter type
        if lowcut is not None and highcut is not None:
            btype = 'band'
            Wn = [lowcut, highcut]
        elif lowcut is not None:
            btype = 'high'
            Wn = lowcut
        elif highcut is not None:
            btype = 'low'
            Wn = highcut
        else:
            return signal  # No filtering
        
        # Design filter
        sos = scipy_signal.butter(order, Wn, btype=btype, 
                                 fs=self.sampling_rate, output='sos')
        
        # Apply filter
        if zero_phase:
            filtered = scipy_signal.sosfiltfilt(sos, signal)
        else:
            filtered = scipy_signal.sosfilt(sos, signal)
            
        return filtered
    
    def fir_filter(self,
                   signal: np.ndarray,
                   lowcut: Optional[float] = None,
                   highcut: Optional[float] = None,
                   numtaps: Optional[int] = None,
                   window: str = 'hamming') -> np.ndarray:
        """
        Apply FIR filter using window method.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        lowcut : float or None
            High-pass cutoff frequency in Hz
        highcut : float or None
            Low-pass cutoff frequency in Hz
        numtaps : int or None
            Number of taps (filter order + 1). If None, automatically calculated.
        window : str
            Window type ('hamming', 'hann', 'blackman', etc.)
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        # Check minimum signal length
        if len(signal) < 50:
            raise ValueError(f"Signal too short for FIR filtering. "
                           f"Signal length: {len(signal)}, minimum required: 50 samples. "
                           f"Consider using Butterworth filter instead.")
        
        # Validate cutoff frequencies (must be < Nyquist frequency)
        nyquist = self.sampling_rate / 2.0
        
        if lowcut is not None and lowcut >= nyquist:
            lowcut = nyquist * 0.99
        if highcut is not None and highcut >= nyquist:
            highcut = nyquist * 0.99
        
        # Ensure lowcut < highcut for bandpass
        if lowcut is not None and highcut is not None and lowcut >= highcut:
            lowcut = highcut * 0.5
        
        # Determine numtaps if not specified
        if numtaps is None:
            # Rule of thumb: numtaps â‰ˆ 3 * (fs / transition_width)
            if lowcut is not None:
                transition_width = min(lowcut * 0.25, 10)
            elif highcut is not None:
                transition_width = min((self.sampling_rate / 2 - highcut) * 0.25, 10)
            else:
                transition_width = 10
            numtaps = int(3 * (self.sampling_rate / transition_width))
            if numtaps % 2 == 0:
                numtaps += 1  # Make odd
            
            # CRITICAL: Limit numtaps to prevent filtfilt padding errors
            # filtfilt needs padlen = 3 * max(len(b), len(a))
            # For safety, numtaps should be much smaller than signal length
            max_numtaps = max(101, int(len(signal) * 0.3))  # At most 30% of signal length
            if numtaps > max_numtaps:
                numtaps = max_numtaps
                if numtaps % 2 == 0:
                    numtaps -= 1  # Keep it odd
        
        # Determine filter type
        if lowcut is not None and highcut is not None:
            # Bandpass
            taps = scipy_signal.firwin(numtaps, [lowcut, highcut], 
                                      pass_zero=False, fs=self.sampling_rate,
                                      window=window)
        elif lowcut is not None:
            # Highpass
            taps = scipy_signal.firwin(numtaps, lowcut, 
                                      pass_zero=False, fs=self.sampling_rate,
                                      window=window)
        elif highcut is not None:
            # Lowpass
            taps = scipy_signal.firwin(numtaps, highcut, 
                                      pass_zero=True, fs=self.sampling_rate,
                                      window=window)
        else:
            return signal  # No filtering
        
        # Apply filter (zero-phase by default using filtfilt)
        filtered = scipy_signal.filtfilt(taps, 1.0, signal)
        
        return filtered
    
    def notch_filter(self,
                    signal: np.ndarray,
                    freq: float = 50.0,
                    quality: float = 30.0,
                    zero_phase: bool = True) -> np.ndarray:
        """
        Apply notch filter (e.g., for 50/60 Hz line noise removal).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        freq : float
            Frequency to remove in Hz (typically 50 or 60)
        quality : float
            Quality factor (higher = narrower notch)
        zero_phase : bool
            If True, apply filter forwards and backwards
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        # Design notch filter
        b, a = scipy_signal.iirnotch(freq, quality, fs=self.sampling_rate)
        
        # Apply filter
        if zero_phase:
            filtered = scipy_signal.filtfilt(b, a, signal)
        else:
            filtered = scipy_signal.lfilter(b, a, signal)
            
        return filtered
    
    def moving_average_filter(self,
                             signal: np.ndarray,
                             window_size: int = 5) -> np.ndarray:
        """
        Apply moving average filter (simple smoothing).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        window_size : int
            Size of moving average window (number of samples)
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        kernel = np.ones(window_size) / window_size
        filtered = scipy_signal.filtfilt(kernel, 1.0, signal)
        return filtered
    
    def savitzky_golay_filter(self,
                             signal: np.ndarray,
                             window_length: int = 11,
                             polyorder: int = 3) -> np.ndarray:
        """
        Apply Savitzky-Golay filter (polynomial smoothing).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        window_length : int
            Length of filter window (must be odd)
        polyorder : int
            Order of polynomial used to fit samples
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        if window_length % 2 == 0:
            window_length += 1
            
        filtered = scipy_signal.savgol_filter(signal, window_length, polyorder)
        return filtered
    
    def wavelet_denoise(self,
                       signal: np.ndarray,
                       wavelet: str = 'db6',
                       level: Optional[int] = None,
                       threshold_method: str = 'universal',
                       threshold_scale: float = 1.0,
                       mode: str = 'soft') -> np.ndarray:
        """
        Denoise signal using Discrete Wavelet Transform (DWT) thresholding.

        Unlike frequency-domain filters, wavelet denoising operates in
        time-frequency space simultaneously, making it well suited to
        brief transient signals such as evoked potentials.

        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        wavelet : str
            Wavelet family. Recommended for biomedical signals:
            'db4', 'db6', 'db8'  - Daubechies (asymmetric, compact)
            'sym4', 'sym6', 'sym8' - Symlets (near-symmetric, good shape preservation)
            'coif3'               - Coiflets (symmetric, good for smooth signals)
        level : int or None
            Decomposition level. If None, automatically chosen based on
            signal length (floor(log2(N)) - 2, capped at 6).
        threshold_method : str
            How to calculate the denoising threshold:
            'universal' - VisuShrink: sigma * sqrt(2 * log(N))
                          Conservative — good default for EPs.
            'bayes'     - BayesShrink: per-level adaptive threshold.
                          More aggressive — better for very noisy signals.
            'manual'    - threshold_scale used directly as threshold value.
        threshold_scale : float
            Multiplier on the computed threshold (universal/bayes) or
            the absolute threshold value (manual). Default 1.0.
        mode : str
            Thresholding mode:
            'soft' - shrinks coefficients toward zero (smoother result,
                     recommended for EP shape preservation)
            'hard' - zeroes coefficients below threshold (sharper edges,
                     may introduce ringing)

        Returns:
        --------
        filtered : np.ndarray
            Denoised signal, same length as input.
        """
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets (pywt) is required for wavelet denoising. "
                "Install it with: pip install PyWavelets"
            )

        # Auto-select decomposition level
        if level is None:
            max_level = pywt.dwt_max_level(len(signal), wavelet)
            level = min(max(3, max_level - 2), 6)

        # Decompose signal into wavelet coefficients
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Estimate noise standard deviation from finest detail level
        # using the robust MAD estimator (median absolute deviation)
        # This is standard practice (Donoho & Johnstone, 1994)
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # MAD estimator
        N = len(signal)

        # Threshold and denoise each detail level
        denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients unchanged

        for i, detail in enumerate(coeffs[1:], 1):
            n_level = len(detail)

            if threshold_method == 'universal':
                # VisuShrink universal threshold
                threshold = threshold_scale * sigma * np.sqrt(2 * np.log(N))

            elif threshold_method == 'bayes':
                # BayesShrink: per-level adaptive threshold
                # Estimates signal variance at each level
                signal_var = max(np.var(detail) - sigma**2, 1e-10)
                threshold = threshold_scale * (sigma**2 / np.sqrt(signal_var))

            elif threshold_method == 'manual':
                threshold = threshold_scale

            else:
                threshold = threshold_scale * sigma * np.sqrt(2 * np.log(N))

            # Apply thresholding
            denoised_detail = pywt.threshold(detail, threshold, mode=mode)
            denoised_coeffs.append(denoised_detail)

        # Reconstruct signal from thresholded coefficients
        filtered = pywt.waverec(denoised_coeffs, wavelet)

        # Trim or pad to original length (waverec can add 1 extra sample)
        if len(filtered) > len(signal):
            filtered = filtered[:len(signal)]
        elif len(filtered) < len(signal):
            filtered = np.pad(filtered, (0, len(signal) - len(filtered)), mode='edge')

        return filtered

    def apply_filter_cascade(self,
                            signal: np.ndarray,
                            filter_params: Dict) -> np.ndarray:
        """
        Apply cascade of filters based on parameter dictionary.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
        filter_params : dict
            Dictionary with filter parameters:
            - filter_type: 'butterworth', 'fir', 'notch', etc.
            - lowcut: high-pass cutoff
            - highcut: low-pass cutoff
            - order: filter order
            - notch_freq: notch frequency
            - notch_enabled: bool
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        filtered = signal.copy()
        
        try:
            # Apply notch filter if enabled
            if filter_params.get('notch_enabled', False):
                notch_freq = filter_params.get('notch_freq', 50.0)
                filtered = self.notch_filter(filtered, freq=notch_freq)
            
            # Apply main filter
            filter_type = filter_params.get('filter_type', 'butterworth')
            lowcut = filter_params.get('lowcut', None)
            highcut = filter_params.get('highcut', None)
            order = filter_params.get('order', 4)
            
            if filter_type == 'butterworth':
                filtered = self.butterworth_filter(filtered, lowcut, highcut, 
                                                  order=order, zero_phase=True)
            elif filter_type == 'butterworth_single_pass':
                filtered = self.butterworth_filter(filtered, lowcut, highcut, 
                                                  order=order, zero_phase=False)
            elif filter_type == 'fir_hamming':
                filtered = self.fir_filter(filtered, lowcut, highcut, window='hamming')
            elif filter_type == 'fir_hann':
                filtered = self.fir_filter(filtered, lowcut, highcut, window='hann')
            elif filter_type == 'fir_blackman':
                filtered = self.fir_filter(filtered, lowcut, highcut, window='blackman')
            elif filter_type == 'moving_average':
                window_size = filter_params.get('window_size', 5)
                filtered = self.moving_average_filter(filtered, window_size)
            elif filter_type == 'savitzky_golay':
                window_length = filter_params.get('window_length', 11)
                polyorder = filter_params.get('polyorder', 3)
                filtered = self.savitzky_golay_filter(filtered, window_length, polyorder)
            elif filter_type == 'wavelet_denoise':
                filtered = self.wavelet_denoise(
                    filtered,
                    wavelet=filter_params.get('wavelet_family', 'db6'),
                    level=filter_params.get('wavelet_level', None),
                    threshold_method=filter_params.get('threshold_method', 'universal'),
                    threshold_scale=filter_params.get('threshold_scale', 1.0),
                    mode=filter_params.get('wavelet_mode', 'soft')
                )
        
        except ValueError as e:
            # Provide helpful error message
            error_msg = str(e)
            if 'padlen' in error_msg or 'greater than' in error_msg:
                raise ValueError(
                    f"Signal too short for {filter_type} filter. "
                    f"Signal length: {len(signal)} samples. "
                    f"Solution: Use Butterworth filter instead, or increase signal duration."
                ) from e
            else:
                # Re-raise other ValueErrors with original message
                raise
        
        return filtered
    
    def get_frequency_response(self,
                              filter_params: Dict,
                              n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response of specified filter.
        
        Parameters:
        -----------
        filter_params : dict
            Filter parameters
        n_points : int
            Number of points for frequency response
            
        Returns:
        --------
        frequencies : np.ndarray
            Frequency vector in Hz
        response : np.ndarray
            Magnitude response in dB
        """
        filter_type = filter_params.get('filter_type', 'butterworth')
        lowcut = filter_params.get('lowcut', None)
        highcut = filter_params.get('highcut', None)
        order = filter_params.get('order', 4)
        
        # Determine filter type for scipy
        if lowcut is not None and highcut is not None:
            btype = 'band'
            Wn = [lowcut, highcut]
        elif lowcut is not None:
            btype = 'high'
            Wn = lowcut
        elif highcut is not None:
            btype = 'low'
            Wn = highcut
        else:
            # No filter
            freqs = np.linspace(0, self.sampling_rate / 2, n_points)
            return freqs, np.zeros(n_points)
        
        if 'butterworth' in filter_type:
            # Design filter
            sos = scipy_signal.butter(order, Wn, btype=btype, 
                                     fs=self.sampling_rate, output='sos')
            
            # Calculate frequency response
            freqs, response = scipy_signal.sosfreqz(sos, worN=n_points, 
                                                    fs=self.sampling_rate)
            
        elif 'fir' in filter_type:
            # Simplified FIR response
            window = filter_type.split('_')[1] if '_' in filter_type else 'hamming'
            
            if lowcut is not None and highcut is not None:
                taps = scipy_signal.firwin(101, [lowcut, highcut], 
                                          pass_zero=False, fs=self.sampling_rate,
                                          window=window)
            elif lowcut is not None:
                taps = scipy_signal.firwin(101, lowcut, 
                                          pass_zero=False, fs=self.sampling_rate,
                                          window=window)
            else:
                taps = scipy_signal.firwin(101, highcut, 
                                          pass_zero=True, fs=self.sampling_rate,
                                          window=window)
            
            freqs, response = scipy_signal.freqz(taps, worN=n_points, 
                                                fs=self.sampling_rate)
        else:
            # Default to Butterworth
            sos = scipy_signal.butter(order, Wn, btype=btype, 
                                     fs=self.sampling_rate, output='sos')
            freqs, response = scipy_signal.sosfreqz(sos, worN=n_points, 
                                                    fs=self.sampling_rate)
        
        # Convert to dB
        response_db = 20 * np.log10(np.abs(response) + 1e-10)

        return freqs, response_db

    def get_wavelet_frequency_bands(self,
                                    wavelet: str = 'db6',
                                    level: int = 5) -> dict:
        """
        Return approximate frequency bands for each wavelet decomposition level.
        Useful for understanding which frequencies each level captures.

        Parameters:
        -----------
        wavelet : str
            Wavelet family name
        level : int
            Number of decomposition levels

        Returns:
        --------
        dict : level -> (f_low, f_high) in Hz
        """
        nyquist = self.sampling_rate / 2.0
        bands = {}
        f_high = nyquist
        for lev in range(1, level + 1):
            f_low = f_high / 2.0
            bands[f'Detail level {lev}'] = (round(f_low, 1), round(f_high, 1))
            f_high = f_low
        bands[f'Approx level {level}'] = (0.0, round(f_high, 1))
        return bands
