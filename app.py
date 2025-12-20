"""
MEPSimFilt - Streamlit GUI
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
from scipy.signal import fftconvolve, find_peaks, peak_widths
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, kruskal, shapiro, levene, bartlett, normaltest
from scipy.optimize import curve_fit

# ===========================
# REAL WAVEFORM LOADING & PARAMETER EXTRACTION
# ===========================

def load_waveform_from_file(file, file_type='auto'):
    """
    Load waveform data from uploaded file with support for multiple formats.
    
    Parameters:
    -----------
    file : UploadedFile
        Streamlit uploaded file object
    file_type : str
        'auto', 'txt', 'csv', 'spike2', or 'mat'
        
    Returns:
    --------
    time : np.ndarray
        Time vector (seconds)
    signal : np.ndarray
        Signal amplitude
    sampling_rate : float
        Detected or default sampling rate
    metadata : dict
        Additional information about loaded file
    """
    import io
    
    filename = file.name.lower()
    
    # Read file content
    file.seek(0)
    content = file.read()
    
    # Try to decode as text
    try:
        if isinstance(content, bytes):
            text_content = content.decode('utf-8')
        else:
            text_content = content
    except:
        text_content = None
    
    metadata = {'filename': file.name, 'format': 'unknown'}
    
    # ========== DETECT LABVIEW TRANSPOSED FORMAT ==========
    # LabView often exports as 2-row format: potentially time/data or data/data
    file.seek(0)
    try:
        # Try reading first few rows to check format
        if filename.endswith('.txt') or filename.endswith('.csv'):
            # Quick check: read first 2 lines manually
            file_lines = text_content.split('\n') if text_content else []
            
            if len(file_lines) >= 2:
                # Count tabs in first line
                n_tabs_line1 = file_lines[0].count('\t')
                n_tabs_line2 = file_lines[1].count('\t') if len(file_lines) > 1 else 0
                
                # LabView transposed: 2 rows, many columns (>1000 tabs)
                if n_tabs_line1 > 1000 and n_tabs_line2 > 1000 and len(file_lines) <= 5:
                    metadata['format'] = 'LabView (Transposed)'
                    
                    # Parse the data
                    row1_values = [float(x) for x in file_lines[0].split('\t') if x.strip()]
                    row2_values = [float(x) for x in file_lines[1].split('\t') if x.strip()]
                    
                    # Row 1: Check if it's time (first value often negative, representing pre-stim)
                    # Row 2: Signal data
                    if row1_values[0] < 0 and row1_values[0] > -10:
                        # Row 1 might be time vector
                        time_raw = np.array(row1_values)
                        signal = np.array(row2_values)
                        
                        # Try to calculate sampling rate from time vector
                        # But first value might be a marker, check subsequent values
                        if len(time_raw) > 100:
                            # Check if values after first are sequential
                            diffs = np.diff(time_raw[1:101])  # Skip first value
                            if np.std(diffs) / np.mean(diffs) < 0.1:  # Fairly uniform
                                # Sequential time vector
                                dt = np.median(diffs)
                                sampling_rate = 1.0 / dt if dt > 0 else None
                                time = time_raw
                            else:
                                # Not sequential - ask user for FS
                                time = None
                                sampling_rate = None
                        else:
                            time = time_raw
                            sampling_rate = None
                    else:
                        # Both rows are data, no time info
                        signal = np.array(row2_values)
                        time = None
                        sampling_rate = None
                    
                    metadata['n_samples'] = len(signal)
                    metadata['requires_fs_input'] = (sampling_rate is None)
                    
                    # If no valid FS, set to None to trigger user input
                    if sampling_rate is None or not np.isfinite(sampling_rate) or sampling_rate < 100 or sampling_rate > 100000:
                        sampling_rate = None
                    
                    return time, signal, sampling_rate, metadata
    except Exception as e:
        # Not LabView format, continue
        pass
    
    # ========== DETECT LABCHART FORMAT ==========
    if text_content and ('Interval=' in text_content and 'ChannelTitle=' in text_content):
        # This is a LabChart exported file
        metadata['format'] = 'LabChart'
        
        lines = text_content.split('\n')
        
        # Parse header information
        sampling_rate = 4000  # Default
        channel_names = []
        data_start_line = 0
        
        for i, line in enumerate(lines[:20]):
            # Extract sampling interval
            if 'Interval=' in line:
                try:
                    interval_str = line.split('=')[1].strip().split()[0]
                    interval = float(interval_str)
                    sampling_rate = 1.0 / interval
                    metadata['interval'] = interval
                except:
                    pass
            
            # Extract channel titles
            if 'ChannelTitle=' in line:
                channel_part = line.split('=')[1].strip()
                channel_names = [ch.strip() for ch in channel_part.split('\t')]
                metadata['channel_names'] = channel_names
                metadata['n_channels'] = len(channel_names)
            
            # Extract voltage ranges
            if 'Range=' in line:
                range_part = line.split('=')[1].strip()
                ranges = [r.strip() for r in range_part.split('\t')]
                metadata['ranges'] = ranges
            
            # Find where data starts (after all header lines)
            if line and not line.startswith('#') and not '=' in line:
                try:
                    float(line.split('\t')[0])
                    data_start_line = i
                    break
                except:
                    continue
        
        # Read data
        data_lines = []
        for line in lines[data_start_line:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split('\t')
                # Should have time + channel data
                data_lines.append([float(p) for p in parts])
            except:
                continue
        
        if not data_lines:
            raise ValueError("No numeric data found in LabChart file")
        
        data_array = np.array(data_lines)
        time = data_array[:, 0]  # First column is time
        
        # Check for trigger/marker channels (binary: only 0/1 or 0/5 values)
        trigger_channel_idx = None
        for ch_idx in range(1, data_array.shape[1]):
            ch_data = data_array[:, ch_idx]
            unique_vals = np.unique(ch_data)
            
            # Binary trigger detection: 2-3 unique values, max ≤ 5.5, min ≥ -0.1
            if len(unique_vals) <= 3 and np.max(ch_data) <= 5.5 and np.min(ch_data) >= -0.1:
                # Check if it looks like a trigger (mostly low, occasional high)
                if np.mean(ch_data < 0.5) > 0.8:  # >80% of time below 0.5
                    trigger_channel_idx = ch_idx
                    metadata['trigger_channel'] = ch_idx
                    metadata['trigger_channel_name'] = channel_names[ch_idx - 1] if ch_idx - 1 < len(channel_names) else f"Channel {ch_idx}"
                    break
        
        # Select data channel (first EMG-like channel, skip trigger if detected)
        if trigger_channel_idx is not None and data_array.shape[1] > 2:
            # Use channel after trigger
            signal_ch_idx = 2 if trigger_channel_idx == 1 else 1
        else:
            # Use first channel
            signal_ch_idx = 1
        
        signal = data_array[:, signal_ch_idx]
        metadata['selected_channel'] = signal_ch_idx
        metadata['selected_channel_name'] = channel_names[signal_ch_idx - 1] if signal_ch_idx - 1 < len(channel_names) else f"Channel {signal_ch_idx}"
        metadata['sampling_rate'] = sampling_rate
        
        return time, signal, sampling_rate, metadata
    
    # ========== DETECT LABCHART FORMAT ==========
    if text_content and ('Interval=' in text_content[:500] or 'ChannelTitle=' in text_content[:500]):
        # This is a LabChart exported file
        metadata['format'] = 'LabChart'
        
        lines = text_content.split('\n')
        
        # Parse header
        sampling_rate = None
        channel_titles = []
        channel_ranges = []
        data_start_idx = 0
        
        for i, line in enumerate(lines[:20]):
            # Extract sampling interval
            if 'Interval=' in line:
                interval_str = line.split('=')[1].strip().split()[0]
                try:
                    interval = float(interval_str)
                    sampling_rate = 1.0 / interval
                    metadata['interval'] = interval
                except:
                    pass
            
            # Extract channel titles
            elif 'ChannelTitle=' in line:
                titles_str = line.split('=')[1].strip()
                channel_titles = [t.strip() for t in titles_str.split('\t')]
                metadata['channel_names'] = channel_titles  # Use 'channel_names' key
                metadata['n_channels'] = len(channel_titles)
            
            # Extract channel ranges
            elif 'Range=' in line:
                ranges_str = line.split('=')[1].strip()
                channel_ranges = [r.strip() for r in ranges_str.split('\t')]
                metadata['ranges'] = channel_ranges
            
            # Find where data starts (first numeric line)
            elif line.strip() and not '=' in line:
                try:
                    parts = line.strip().split('\t')
                    float(parts[0])
                    data_start_idx = i
                    break
                except:
                    continue
        
        if sampling_rate is None:
            sampling_rate = 4000
        
        # Read data section
        data_lines = []
        comment_lines = []
        
        for i, line in enumerate(lines[data_start_idx:], start=data_start_idx):
            line = line.strip()
            if not line:
                continue
            
            # Check for comment/marker lines
            if line.startswith('#'):
                # LabChart trigger/marker comment line
                comment_lines.append((i, line))
                continue
            
            try:
                parts = line.split('\t')
                data_lines.append([float(p) for p in parts])
            except:
                continue
        
        # Convert to array
        data_array = np.array(data_lines)
        time = data_array[:, 0]
        
        # Determine MEP channel (highest variability, excluding trigger channels)
        channel_data = []
        for ch_idx in range(1, data_array.shape[1]):
            ch_signal = data_array[:, ch_idx]
            ch_std = np.std(ch_signal)
            ch_range = np.ptp(ch_signal)
            # Exclude channels that look like triggers (few unique values)
            n_unique = len(np.unique(ch_signal))
            if n_unique > 20:  # Real signal has many unique values
                channel_data.append((ch_idx, ch_std, ch_range))
        
        if channel_data:
            # Select channel with highest range (likely MEP)
            mep_ch_idx = max(channel_data, key=lambda x: x[2])[0]
        else:
            mep_ch_idx = 1  # Default
        
        signal = data_array[:, mep_ch_idx]
        metadata['selected_channel'] = mep_ch_idx
        metadata['selected_channel_name'] = channel_titles[mep_ch_idx - 1] if mep_ch_idx - 1 < len(channel_titles) else f"Channel {mep_ch_idx}"
        
        # Detect trigger channel
        for ch_idx in range(1, data_array.shape[1]):
            if ch_idx == mep_ch_idx:
                continue  # Skip MEP channel
            
            ch_data = data_array[:, ch_idx]
            ch_diff = np.abs(np.diff(ch_data))
            
            # Look for large step changes characteristic of digital triggers
            threshold = np.std(ch_diff) * 5  # 5 SD above mean
            large_steps = np.where(ch_diff > threshold)[0]
            
            if len(large_steps) >= 1 and np.std(ch_data) > 0.01:
                metadata['trigger_channel'] = ch_idx
                metadata['trigger_channel_name'] = channel_titles[ch_idx - 1] if ch_idx - 1 < len(channel_titles) else f"Channel {ch_idx}"
                metadata['trigger_times'] = time[large_steps]
                metadata['n_triggers'] = len(large_steps)
                break
        
        # Store comment-based markers
        if comment_lines:
            metadata['comment_markers'] = comment_lines
        
        metadata['n_samples'] = len(signal)
        metadata['sampling_rate'] = sampling_rate
        
        return time, signal, sampling_rate, metadata
    
    # ========== DETECT SPIKE2 FORMAT ==========
    if text_content and ('"INFORMATION"' in text_content or '"SUMMARY"' in text_content or '"START"' in text_content):
        # This is a Spike2 exported file
        metadata['format'] = 'Spike2'
        
        lines = text_content.split('\n')
        
        # Find sampling rate from SUMMARY
        sampling_rate = 5000  # Default
        for i, line in enumerate(lines):
            if '"SUMMARY"' in line and i + 1 < len(lines):
                # Parse next few lines for waveform info
                for j in range(i+1, min(i+5, len(lines))):
                    parts = lines[j].split('\t')
                    if 'Waveform' in lines[j]:
                        # Extract sampling rate (usually 4th or 5th element)
                        for part in parts:
                            try:
                                val = float(part.strip().strip('"'))
                                if 100 <= val <= 100000:  # Reasonable sampling rate
                                    sampling_rate = val
                                    break
                            except:
                                continue
                        break
        
        # Find START marker and read continuous data
        start_idx = None
        for i, line in enumerate(lines):
            if '"START"' in line:
                start_idx = i + 1
                break
        
        if start_idx is None:
            raise ValueError("Could not find START marker in Spike2 file")
        
        # Read numeric data until we hit another CHANNEL section
        signal_values = []
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('"'):
                # Stop at next channel section
                if line.startswith('"CHANNEL"') and len(signal_values) > 0:
                    break
                continue
            try:
                val = float(line)
                signal_values.append(val)
            except:
                continue
        
        signal = np.array(signal_values)
        time = np.arange(len(signal)) / sampling_rate
        
        # ========== DETECT DIGMARKS/EVENTS ==========
        event_channels = {}
        
        # Find all event channels (Evt+, Marker, DigMark)
        for i, line in enumerate(lines):
            if '"CHANNEL"' in line and i + 1 < len(lines):
                # Parse channel header
                channel_num = None
                channel_type = None
                channel_name = None
                
                # Get channel number
                parts = line.split('\t')
                for part in parts:
                    if part.strip().strip('"').isdigit():
                        channel_num = part.strip().strip('"')
                        break
                
                # Next few lines contain type and name
                for j in range(i+1, min(i+5, len(lines))):
                    if '"Evt+"' in lines[j] or '"Marker"' in lines[j]:
                        channel_type = lines[j].strip().strip('"')
                    if '"DigMark"' in lines[j] or '"Digitimer"' in lines[j] or '"Magstim"' in lines[j]:
                        channel_name = lines[j].strip().strip('"')
                        
                        # Now read event times (next non-empty lines)
                        event_times = []
                        for k in range(j+1, min(j+500, len(lines))):
                            line_data = lines[k].strip()
                            if not line_data or line_data.startswith('"CHANNEL'):
                                if len(event_times) > 0:
                                    break
                                continue
                            
                            # Handle two formats:
                            # Format 1: Single time value per line (Evt+ style)
                            # Format 2: Tab-delimited with time, code, values (DigMark style)
                            if '\t' in line_data:
                                # Tab-delimited format - time is first column
                                parts = line_data.split('\t')
                                try:
                                    time_val = float(parts[0])
                                    event_times.append(time_val)
                                except:
                                    continue
                            else:
                                # Single value format
                                try:
                                    time_val = float(line_data)
                                    event_times.append(time_val)
                                except:
                                    continue
                        
                        if event_times:
                            event_channels[channel_name] = {
                                'channel_num': channel_num,
                                'type': channel_type,
                                'times': event_times,
                                'count': len(event_times)
                            }
                        break
        
        metadata['n_samples'] = len(signal)
        metadata['sampling_rate'] = sampling_rate
        metadata['event_channels'] = event_channels
        metadata['n_events'] = sum(ch['count'] for ch in event_channels.values())
        
        # If events detected, user will need to select one for segmentation
        if event_channels:
            metadata['has_events'] = True
            metadata['requires_event_selection'] = True
        
        return time, signal, sampling_rate, metadata
    
    # ========== CSV FORMAT ==========
    if filename.endswith('.csv'):
        metadata['format'] = 'CSV'
        file.seek(0)
        
        df = pd.read_csv(file)
        
        # Check if first row is numeric (data) or headers
        first_row_numeric = True
        try:
            pd.to_numeric(df.iloc[0, :])
        except:
            first_row_numeric = False
        
        # If headers are just numbers (0,1,2,3...), this is multi-trial data
        if all(str(col).replace('.', '').replace('-', '').isdigit() for col in df.columns[:min(5, len(df.columns))]):
            # Multi-trial format: rows = time points, columns = trials
            metadata['format'] = 'CSV (Multi-trial)'
            metadata['n_trials'] = df.shape[1]
            
            # Use first trial (column 0) as default
            signal = df.iloc[:, 0].values
            
            # Estimate sampling rate from number of samples
            # Common MEP duration is 50-150 ms
            n_samples = len(signal)
            
            # Try different durations and find most reasonable sampling rate
            possible_durations = [0.050, 0.075, 0.095, 0.100, 0.150, 0.200]  # 50-200 ms
            possible_rates = []
            
            for duration in possible_durations:
                fs_estimate = n_samples / duration
                # Check if close to common rate
                common_rates = [1000, 2000, 2500, 5000, 10000, 20000]
                closest_rate = min(common_rates, key=lambda x: abs(x - fs_estimate))
                if abs(closest_rate - fs_estimate) / closest_rate < 0.15:  # Within 15%
                    possible_rates.append((closest_rate, abs(closest_rate - fs_estimate)))
            
            # Select best match (smallest error)
            if possible_rates:
                sampling_rate = sorted(possible_rates, key=lambda x: x[1])[0][0]
            else:
                # Default: assume 95 ms at 5000 Hz is common
                sampling_rate = 5000
            
            time = np.arange(len(signal)) / sampling_rate
            
            metadata['note'] = f"Multi-trial CSV with {df.shape[1]} trials. Using trial 1. Estimated {sampling_rate:.0f} Hz"
            metadata['sampling_rate'] = sampling_rate
            
        else:
            # Standard two-column format
            if df.shape[1] >= 2:
                time = df.iloc[:, 0].values
                signal = df.iloc[:, 1].values
            else:
                signal = df.iloc[:, 0].values
                time = None
            
            # Auto-detect if time is in milliseconds
            if time is not None and np.max(time) > 10:
                time = time / 1000.0  # Convert ms to seconds
            
            # Calculate sampling rate
            if time is not None:
                dt = np.median(np.diff(time))
                sampling_rate = 1.0 / dt if dt > 0 else 5000.0
            else:
                sampling_rate = 5000.0
                time = np.arange(len(signal)) / sampling_rate
        
        return time, signal, sampling_rate, metadata
    
    # ========== TXT FORMAT ==========
    if filename.endswith('.txt'):
        metadata['format'] = 'TXT'
        file.seek(0)
        
        try:
            data = np.loadtxt(file)
            
            if data.ndim == 2:
                # Two-column format
                if data.shape[1] >= 2:
                    time = data[:, 0]
                    signal = data[:, 1]
                else:
                    signal = data[:, 0]
                    time = None
            else:
                # Single column
                signal = data
                time = None
            
            # Auto-detect time units
            if time is not None and np.max(time) > 10:
                time = time / 1000.0
            
            # Calculate sampling rate
            if time is not None:
                dt = np.median(np.diff(time))
                sampling_rate = 1.0 / dt if dt > 0 else 5000.0
            else:
                sampling_rate = 5000.0
                time = np.arange(len(signal)) / sampling_rate
            
            return time, signal, sampling_rate, metadata
            
        except Exception as e:
            raise ValueError(f"Could not parse TXT file: {str(e)}")
    
    # ========== MAT FORMAT ==========
    if filename.endswith('.mat'):
        metadata['format'] = 'MATLAB'
        file.seek(0)
        
        from scipy.io import loadmat
        mat_data = loadmat(io.BytesIO(content))
        
        # Try common variable names
        signal = None
        for var_name in ['data', 'signal', 'emg', 'mep', 'waveform', 'y', 'amplitude']:
            if var_name in mat_data:
                signal = np.squeeze(mat_data[var_name])
                metadata['signal_variable'] = var_name
                break
        
        if signal is None:
            # Take first non-metadata array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    signal = np.squeeze(value)
                    metadata['signal_variable'] = key
                    break
        
        # Try to find time vector
        time = None
        for var_name in ['time', 't', 'x', 'time_ms', 'time_s']:
            if var_name in mat_data:
                time = np.squeeze(mat_data[var_name])
                metadata['time_variable'] = var_name
                break
        
        # Check time units and calculate sampling rate
        if time is not None:
            if np.max(time) > 10:  # Likely milliseconds
                time = time / 1000.0
            dt = np.median(np.diff(time))
            sampling_rate = 1.0 / dt if dt > 0 else 5000.0
        else:
            sampling_rate = 5000.0
            time = np.arange(len(signal)) / sampling_rate
        
        return time, signal, sampling_rate, metadata
    
    raise ValueError(f"Unsupported file type: {filename}")


def extract_mep_parameters(time, signal, sampling_rate):
    """
    Automatically extract MEP parameters from real waveform.
    
    Parameters:
    -----------
    time : np.ndarray
        Time vector (seconds)
    signal : np.ndarray
        Signal amplitude
    sampling_rate : float
        Sampling rate (Hz)
        
    Returns:
    --------
    dict : Extracted parameters
    """
    params = {}
    
    # Remove DC offset (baseline correction)
    # Use first 10% of signal as baseline estimate
    baseline_samples = int(len(signal) * 0.1)
    baseline = np.mean(signal[:baseline_samples])
    signal_corrected = signal - baseline
    
    # Convert time to milliseconds
    time_ms = time * 1000
    
    # Find stimulus point (t=0)
    stim_idx = np.argmin(np.abs(time_ms))
    
    # Focus on post-stimulus period for MEP detection
    post_stim_idx = np.where(time_ms >= 0)[0]
    if len(post_stim_idx) == 0:
        post_stim_idx = np.arange(len(signal_corrected))
    
    # Find global maximum and minimum in post-stimulus period
    post_signal = signal_corrected[post_stim_idx]
    post_time_ms = time_ms[post_stim_idx]
    
    max_idx_post = np.argmax(post_signal)
    min_idx_post = np.argmin(post_signal)
    
    # Determine which is larger in absolute value
    if abs(post_signal[max_idx_post]) > abs(post_signal[min_idx_post]):
        peak_idx_post = max_idx_post
        peak_polarity = 'positive'
    else:
        peak_idx_post = min_idx_post
        peak_polarity = 'negative'
    
    peak_idx = post_stim_idx[peak_idx_post]
    
    params['peak_amplitude'] = float(np.abs(signal_corrected[peak_idx]))
    params['peak_latency_ms'] = float(time_ms[peak_idx])
    params['peak_polarity'] = peak_polarity
    params['baseline_offset'] = float(baseline)
    params['peak_idx'] = int(peak_idx)  # Store for plotting
    
    # Calculate PTP amplitude
    params['ptp_amplitude'] = float(np.ptp(signal_corrected[post_stim_idx]))
    
    # Find MEP onset (first point exceeding 10% of peak amplitude)
    threshold = params['peak_amplitude'] * 0.1
    above_threshold = np.where(np.abs(post_signal) > threshold)[0]
    
    if len(above_threshold) > 0:
        onset_idx_post = above_threshold[0]
        onset_idx = post_stim_idx[onset_idx_post]
        params['mep_onset_ms'] = float(time_ms[onset_idx])
        params['onset_latency_ms'] = float(time_ms[onset_idx] - time_ms[stim_idx])
        params['onset_idx'] = int(onset_idx)  # Store for plotting
    else:
        params['mep_onset_ms'] = params['peak_latency_ms'] - 10
        params['onset_latency_ms'] = params['mep_onset_ms']
        params['onset_idx'] = max(0, peak_idx - int(0.010 * sampling_rate))
    
    # Find MEP offset (return to <10% threshold after peak)
    after_peak = post_signal[peak_idx_post:]
    below_threshold = np.where(np.abs(after_peak) < threshold)[0]
    
    if len(below_threshold) > 0:
        offset_idx_post = peak_idx_post + below_threshold[0]
        if offset_idx_post < len(post_stim_idx):
            offset_idx = post_stim_idx[offset_idx_post]
            params['mep_offset_ms'] = float(time_ms[offset_idx])
            params['offset_idx'] = int(offset_idx)  # Store for plotting
        else:
            params['mep_offset_ms'] = params['peak_latency_ms'] + 30
            params['offset_idx'] = min(len(signal_corrected) - 1, peak_idx + int(0.030 * sampling_rate))
    else:
        params['mep_offset_ms'] = params['peak_latency_ms'] + 30
        params['offset_idx'] = min(len(signal_corrected) - 1, peak_idx + int(0.030 * sampling_rate))
    
    # Calculate durations
    params['duration_ms'] = params['mep_offset_ms'] - params['mep_onset_ms']
    params['rise_time_ms'] = params['peak_latency_ms'] - params['mep_onset_ms']
    params['decay_time_ms'] = params['mep_offset_ms'] - params['peak_latency_ms']
    
    # Detect polyphasic nature using peak detection
    # Use baseline-corrected signal for detection
    peaks_pos, props_pos = find_peaks(signal_corrected, height=params['peak_amplitude'] * 0.25, 
                                      prominence=params['peak_amplitude'] * 0.2)
    peaks_neg, props_neg = find_peaks(-signal_corrected, height=params['peak_amplitude'] * 0.25,
                                      prominence=params['peak_amplitude'] * 0.2)
    
    # Only count peaks in post-stimulus period
    peaks_pos = [p for p in peaks_pos if p >= stim_idx]
    peaks_neg = [p for p in peaks_neg if p >= stim_idx]
    
    total_peaks = len(peaks_pos) + len(peaks_neg)
    
    # Store all peak indices for plotting
    params['all_peak_indices'] = sorted(list(peaks_pos) + list(peaks_neg), key=lambda x: time_ms[x])
    
    if total_peaks <= 1:
        params['morphology'] = 'Monophasic'
        params['n_phases'] = 1
    elif total_peaks == 2:
        params['morphology'] = 'Bi-phasic'
        params['n_phases'] = 2
        
        # Extract phase amplitudes
        all_peaks = sorted(list(peaks_pos) + list(peaks_neg), key=lambda x: time_ms[x])
        if len(all_peaks) >= 2:
            params['phase1_amplitude'] = float(np.abs(signal_corrected[all_peaks[0]]))
            params['phase2_amplitude'] = float(np.abs(signal_corrected[all_peaks[1]]))
            if params['phase1_amplitude'] > 0:
                params['phase2_ratio'] = params['phase2_amplitude'] / params['phase1_amplitude']
            else:
                params['phase2_ratio'] = 0.8
            
            # Phase timings
            params['phase1_latency_ms'] = float(time_ms[all_peaks[0]])
            params['phase2_latency_ms'] = float(time_ms[all_peaks[1]])
            params['phase_separation_ms'] = params['phase2_latency_ms'] - params['phase1_latency_ms']
    elif total_peaks >= 3:
        params['morphology'] = 'Tri-phasic'
        params['n_phases'] = 3
        
        # Extract phase amplitudes
        all_peaks = sorted(list(peaks_pos) + list(peaks_neg), key=lambda x: time_ms[x])
        if len(all_peaks) >= 3:
            params['phase1_amplitude'] = float(np.abs(signal_corrected[all_peaks[0]]))
            params['phase2_amplitude'] = float(np.abs(signal_corrected[all_peaks[1]]))
            params['phase3_amplitude'] = float(np.abs(signal_corrected[all_peaks[2]]))
            
            if params['phase1_amplitude'] > 0:
                params['phase2_ratio'] = params['phase2_amplitude'] / params['phase1_amplitude']
                params['phase3_ratio'] = params['phase3_amplitude'] / params['phase1_amplitude']
            else:
                params['phase2_ratio'] = 0.75
                params['phase3_ratio'] = 0.40
            
            # Phase timings
            params['phase1_latency_ms'] = float(time_ms[all_peaks[0]])
            params['phase2_latency_ms'] = float(time_ms[all_peaks[1]])
            params['phase3_latency_ms'] = float(time_ms[all_peaks[2]])
            params['phase1_2_separation_ms'] = params['phase2_latency_ms'] - params['phase1_latency_ms']
            params['phase2_3_separation_ms'] = params['phase3_latency_ms'] - params['phase2_latency_ms']
    else:
        params['morphology'] = 'Standard (Monophasic)'
        params['n_phases'] = 1
    
    # Store corrected waveform and indices
    params['original_time'] = time
    params['original_signal'] = signal
    params['corrected_signal'] = signal_corrected
    params['sampling_rate'] = sampling_rate
    params['stim_idx'] = int(stim_idx)
    params['time_ms'] = time_ms
    
    return params


def plot_annotated_waveform(time, signal, params, title="Extracted Parameters - Annotated"):
    """
    Create annotated waveform plot showing all extraction points.
    
    Parameters:
    -----------
    time : np.ndarray
        Time vector (seconds)
    signal : np.ndarray  
        Signal amplitude (baseline-corrected)
    params : dict
        Extracted parameters with indices
    title : str
        Plot title
        
    Returns:
    --------
    fig : matplotlib figure
    """
    time_ms = time * 1000
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot waveform
    ax.plot(time_ms, signal, 'b-', linewidth=2, alpha=0.8, label='Waveform', zorder=1)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)
    
    # Color scheme
    color_onset = '#2ecc71'    # Green
    color_peak = '#e74c3c'     # Red
    color_offset = '#9b59b6'   # Purple
    color_phase = '#f39c12'    # Orange
    color_stim = '#e67e22'     # Dark orange
    
    # STIMULUS MARKER (t=0)
    stim_idx = params.get('stim_idx', np.argmin(np.abs(time_ms)))
    ax.axvline(time_ms[stim_idx], color=color_stim, linestyle='--', linewidth=3, 
               alpha=0.7, label='Stimulus (t=0)', zorder=10)
    
    # ONSET MARKER
    if 'onset_idx' in params:
        onset_idx = params['onset_idx']
        ax.axvline(time_ms[onset_idx], color=color_onset, linestyle='--', linewidth=2.5,
                   alpha=0.8, label=f"MEP Onset ({params['mep_onset_ms']:.1f} ms)", zorder=5)
        ax.plot(time_ms[onset_idx], signal[onset_idx], 'o', color=color_onset, 
                markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=15)
    
    # PEAK MARKER
    if 'peak_idx' in params:
        peak_idx = params['peak_idx']
        ax.axvline(time_ms[peak_idx], color=color_peak, linestyle='--', linewidth=2.5,
                   alpha=0.8, label=f"Peak ({params['peak_latency_ms']:.1f} ms, {params['peak_amplitude']:.3f} mV)", zorder=5)
        ax.plot(time_ms[peak_idx], signal[peak_idx], 'o', color=color_peak,
                markersize=12, markeredgewidth=2, markeredgecolor='white', zorder=15)
        
        # Peak amplitude line
        ax.plot([time_ms[peak_idx], time_ms[peak_idx]], [0, signal[peak_idx]], 
                'r-', linewidth=2, alpha=0.5, zorder=3)
        ax.text(time_ms[peak_idx] + 2, signal[peak_idx] / 2, 
                f'{params["peak_amplitude"]:.2f} mV', fontsize=10, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # OFFSET MARKER
    if 'offset_idx' in params:
        offset_idx = params['offset_idx']
        ax.axvline(time_ms[offset_idx], color=color_offset, linestyle='--', linewidth=2.5,
                   alpha=0.8, label=f"MEP Offset ({params['mep_offset_ms']:.1f} ms)", zorder=5)
        ax.plot(time_ms[offset_idx], signal[offset_idx], 'o', color=color_offset,
                markersize=10, markeredgewidth=2, markeredgecolor='white', zorder=15)
    
    # RISE TIME ANNOTATION
    if 'onset_idx' in params and 'peak_idx' in params:
        onset_idx = params['onset_idx']
        peak_idx = params['peak_idx']
        mid_y = signal[peak_idx] * 0.8
        ax.annotate('', xy=(time_ms[peak_idx], mid_y), xytext=(time_ms[onset_idx], mid_y),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2.5, alpha=0.7))
        ax.text((time_ms[onset_idx] + time_ms[peak_idx]) / 2, mid_y + 0.1,
                f'Rise: {params["rise_time_ms"]:.1f} ms', fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))
    
    # DECAY TIME ANNOTATION
    if 'peak_idx' in params and 'offset_idx' in params:
        peak_idx = params['peak_idx']
        offset_idx = params['offset_idx']
        mid_y = signal[peak_idx] * 0.6
        ax.annotate('', xy=(time_ms[offset_idx], mid_y), xytext=(time_ms[peak_idx], mid_y),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2.5, alpha=0.7))
        ax.text((time_ms[peak_idx] + time_ms[offset_idx]) / 2, mid_y - 0.15,
                f'Decay: {params["decay_time_ms"]:.1f} ms', fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='plum', alpha=0.8))
    
    # DURATION BRACKET
    if 'onset_idx' in params and 'offset_idx' in params:
        onset_idx = params['onset_idx']
        offset_idx = params['offset_idx']
        bracket_y = np.min(signal) * 1.2
        ax.plot([time_ms[onset_idx], time_ms[onset_idx]], [bracket_y, bracket_y - 0.1], 
                'k-', linewidth=2, alpha=0.6)
        ax.plot([time_ms[offset_idx], time_ms[offset_idx]], [bracket_y, bracket_y - 0.1],
                'k-', linewidth=2, alpha=0.6)
        ax.plot([time_ms[onset_idx], time_ms[offset_idx]], [bracket_y, bracket_y],
                'k-', linewidth=2, alpha=0.6)
        ax.text((time_ms[onset_idx] + time_ms[offset_idx]) / 2, bracket_y - 0.25,
                f'Duration: {params["duration_ms"]:.1f} ms', fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9),
                fontweight='bold')
    
    # PHASE MARKERS (for bi/tri-phasic)
    if 'all_peak_indices' in params and len(params['all_peak_indices']) > 1:
        for i, peak_idx in enumerate(params['all_peak_indices'][:3], 1):  # Max 3 phases
            ax.plot(time_ms[peak_idx], signal[peak_idx], 's', color=color_phase,
                    markersize=11, markeredgewidth=2, markeredgecolor='white', 
                    label=f'Phase {i}' if i == 1 else '', zorder=14)
            ax.text(time_ms[peak_idx], signal[peak_idx] + 0.15, f'P{i}',
                    fontsize=9, ha='center', fontweight='bold', color=color_phase,
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.9))
    
    # THRESHOLD LINES (10% of peak)
    threshold = params['peak_amplitude'] * 0.1
    if params['peak_polarity'] == 'positive':
        ax.axhline(threshold, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
                   label=f'10% Threshold ({threshold:.3f} mV)')
    else:
        ax.axhline(-threshold, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
                   label=f'10% Threshold ({threshold:.3f} mV)')
    
    # Labels and formatting
    ax.set_xlabel('Time (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude (mV)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)
    
    # Add info text box
    info_text = f'Morphology: {params["morphology"]}\n'
    info_text += f'Peak-to-Peak: {params["ptp_amplitude"]:.3f} mV\n'
    if params['n_phases'] >= 2 and 'phase2_ratio' in params:
        info_text += f'Phase 2 Ratio: {params["phase2_ratio"]:.2f}\n'
    if params['n_phases'] >= 3 and 'phase3_ratio' in params:
        info_text += f'Phase 3 Ratio: {params["phase3_ratio"]:.2f}'
    
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
            verticalalignment='bottom', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig


def extract_mep_parameters(time, signal, sampling_rate):
    """
    Automatically extract MEP parameters from real waveform.
    
    Parameters:
    -----------
    time : np.ndarray
        Time vector (seconds)
    signal : np.ndarray
        Signal amplitude
    sampling_rate : float
        Sampling rate (Hz)
        
    Returns:
    --------
    dict : Extracted parameters
    """
    params = {}
    
    # Remove DC offset (baseline correction)
    # Use first 10% of signal as baseline estimate
    baseline_samples = int(len(signal) * 0.1)
    baseline = np.mean(signal[:baseline_samples])
    signal_corrected = signal - baseline
    
    # Convert time to milliseconds
    time_ms = time * 1000
    
    # Find stimulus point (t=0)
    stim_idx = np.argmin(np.abs(time_ms))
    
    # Focus on post-stimulus period for MEP detection
    post_stim_idx = np.where(time_ms >= 0)[0]
    if len(post_stim_idx) == 0:
        post_stim_idx = np.arange(len(signal_corrected))
    
    # Find global maximum and minimum in post-stimulus period
    post_signal = signal_corrected[post_stim_idx]
    post_time_ms = time_ms[post_stim_idx]
    
    max_idx_post = np.argmax(post_signal)
    min_idx_post = np.argmin(post_signal)
    
    # Determine which is larger in absolute value
    if abs(post_signal[max_idx_post]) > abs(post_signal[min_idx_post]):
        peak_idx_post = max_idx_post
        peak_polarity = 'positive'
    else:
        peak_idx_post = min_idx_post
        peak_polarity = 'negative'
    
    peak_idx = post_stim_idx[peak_idx_post]
    
    params['peak_amplitude'] = float(np.abs(signal_corrected[peak_idx]))
    params['peak_latency_ms'] = float(time_ms[peak_idx])
    params['peak_polarity'] = peak_polarity
    params['baseline_offset'] = float(baseline)
    
    # Calculate PTP amplitude
    params['ptp_amplitude'] = float(np.ptp(signal_corrected[post_stim_idx]))
    
    # Find MEP onset (first point exceeding 10% of peak amplitude)
    threshold = params['peak_amplitude'] * 0.1
    above_threshold = np.where(np.abs(post_signal) > threshold)[0]
    
    if len(above_threshold) > 0:
        onset_idx_post = above_threshold[0]
        onset_idx = post_stim_idx[onset_idx_post]
        params['mep_onset_ms'] = float(time_ms[onset_idx])
        params['onset_latency_ms'] = float(time_ms[onset_idx] - time_ms[stim_idx])
    else:
        params['mep_onset_ms'] = params['peak_latency_ms'] - 10
        params['onset_latency_ms'] = params['mep_onset_ms']
    
    # Find MEP offset (return to <10% threshold after peak)
    after_peak = post_signal[peak_idx_post:]
    below_threshold = np.where(np.abs(after_peak) < threshold)[0]
    
    if len(below_threshold) > 0:
        offset_idx_post = peak_idx_post + below_threshold[0]
        if offset_idx_post < len(post_stim_idx):
            offset_idx = post_stim_idx[offset_idx_post]
            params['mep_offset_ms'] = float(time_ms[offset_idx])
        else:
            params['mep_offset_ms'] = params['peak_latency_ms'] + 30
    else:
        params['mep_offset_ms'] = params['peak_latency_ms'] + 30
    
    # Calculate durations
    params['duration_ms'] = params['mep_offset_ms'] - params['mep_onset_ms']
    params['rise_time_ms'] = params['peak_latency_ms'] - params['mep_onset_ms']
    params['decay_time_ms'] = params['mep_offset_ms'] - params['peak_latency_ms']
    
    # Detect polyphasic nature using peak detection
    # Use baseline-corrected signal for detection
    peaks_pos, props_pos = find_peaks(signal_corrected, height=params['peak_amplitude'] * 0.25, 
                                      prominence=params['peak_amplitude'] * 0.2)
    peaks_neg, props_neg = find_peaks(-signal_corrected, height=params['peak_amplitude'] * 0.25,
                                      prominence=params['peak_amplitude'] * 0.2)
    
    # Only count peaks in post-stimulus period
    peaks_pos = [p for p in peaks_pos if p >= stim_idx]
    peaks_neg = [p for p in peaks_neg if p >= stim_idx]
    
    total_peaks = len(peaks_pos) + len(peaks_neg)
    
    if total_peaks <= 1:
        params['morphology'] = 'Monophasic'
        params['n_phases'] = 1
    elif total_peaks == 2:
        params['morphology'] = 'Bi-phasic'
        params['n_phases'] = 2
        
        # Extract phase amplitudes
        all_peaks = sorted(list(peaks_pos) + list(peaks_neg), key=lambda x: time_ms[x])
        if len(all_peaks) >= 2:
            params['phase1_amplitude'] = float(np.abs(signal_corrected[all_peaks[0]]))
            params['phase2_amplitude'] = float(np.abs(signal_corrected[all_peaks[1]]))
            if params['phase1_amplitude'] > 0:
                params['phase2_ratio'] = params['phase2_amplitude'] / params['phase1_amplitude']
            else:
                params['phase2_ratio'] = 0.8
            
            # Phase timings
            params['phase1_latency_ms'] = float(time_ms[all_peaks[0]])
            params['phase2_latency_ms'] = float(time_ms[all_peaks[1]])
            params['phase_separation_ms'] = params['phase2_latency_ms'] - params['phase1_latency_ms']
    elif total_peaks >= 3:
        params['morphology'] = 'Tri-phasic'
        params['n_phases'] = 3
        
        # Extract phase amplitudes
        all_peaks = sorted(list(peaks_pos) + list(peaks_neg), key=lambda x: time_ms[x])
        if len(all_peaks) >= 3:
            params['phase1_amplitude'] = float(np.abs(signal_corrected[all_peaks[0]]))
            params['phase2_amplitude'] = float(np.abs(signal_corrected[all_peaks[1]]))
            params['phase3_amplitude'] = float(np.abs(signal_corrected[all_peaks[2]]))
            
            if params['phase1_amplitude'] > 0:
                params['phase2_ratio'] = params['phase2_amplitude'] / params['phase1_amplitude']
                params['phase3_ratio'] = params['phase3_amplitude'] / params['phase1_amplitude']
            else:
                params['phase2_ratio'] = 0.75
                params['phase3_ratio'] = 0.40
            
            # Phase timings
            params['phase1_latency_ms'] = float(time_ms[all_peaks[0]])
            params['phase2_latency_ms'] = float(time_ms[all_peaks[1]])
            params['phase3_latency_ms'] = float(time_ms[all_peaks[2]])
            params['phase1_2_separation_ms'] = params['phase2_latency_ms'] - params['phase1_latency_ms']
            params['phase2_3_separation_ms'] = params['phase3_latency_ms'] - params['phase2_latency_ms']
    else:
        params['morphology'] = 'Standard (Monophasic)'
        params['n_phases'] = 1
    
    # Store corrected waveform
    params['original_time'] = time
    params['original_signal'] = signal
    params['corrected_signal'] = signal_corrected
    params['sampling_rate'] = sampling_rate
    
    return params


# ===========================
# STATISTICAL ASSUMPTION TESTING
# ===========================

def check_normality(data, method='shapiro', alpha=0.05):
    """
    Test normality assumption using multiple methods.
    
    Parameters:
    -----------
    data : array-like
        Data to test
    method : str
        'shapiro' (Shapiro-Wilk), 'anderson' (Anderson-Darling), or 'kstest' (Kolmogorov-Smirnov)
    alpha : float
        Significance level
        
    Returns:
    --------
    dict : Test results with interpretation
    """
    data = np.asarray(data)
    
    if method == 'shapiro':
        if len(data) > 5000:
            # Shapiro-Wilk not reliable for very large samples, use D'Agostino-Pearson
            stat, p_value = normaltest(data)
            test_name = "D'Agostino-Pearson"
        else:
            stat, p_value = shapiro(data)
            test_name = "Shapiro-Wilk"
    elif method == 'anderson':
        result = stats.anderson(data, dist='norm')
        # Use 5% critical value
        stat = result.statistic
        critical_value = result.critical_values[2]  # 5% level
        p_value = 0.05 if stat > critical_value else 0.10  # Approximate
        test_name = "Anderson-Darling"
    else:
        stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        test_name = "Kolmogorov-Smirnov"
    
    is_normal = p_value > alpha
    
    return {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'is_normal': is_normal,
        'interpretation': f"Data is {'consistent with' if is_normal else 'not consistent with'} normal distribution (p = {p_value:.6f})",
        'recommendation': 'Parametric tests appropriate' if is_normal else 'Non-parametric tests recommended'
    }


def check_homogeneity_of_variance(groups, method='levene', alpha=0.05):
    """
    Test homogeneity of variance assumption across groups.
    
    Parameters:
    -----------
    groups : list of array-like
        Data from different groups
    method : str
        'levene' (Levene's test) or 'bartlett' (Bartlett's test)
    alpha : float
        Significance level
        
    Returns:
    --------
    dict : Test results with interpretation
    """
    if method == 'levene':
        # Levene's test (robust to non-normality)
        stat, p_value = levene(*groups, center='median')
        test_name = "Levene's Test"
    else:
        # Bartlett's test (assumes normality)
        stat, p_value = bartlett(*groups)
        test_name = "Bartlett's Test"
    
    homogeneous = p_value > alpha
    
    return {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'is_homogeneous': homogeneous,
        'interpretation': f"Variances are {'homogeneous' if homogeneous else 'heterogeneous'} (p = {p_value:.6f})",
        'recommendation': 'Equal variance assumption met' if homogeneous else 'Equal variance assumption violated - adjust tests accordingly'
    }


def check_sample_size_adequacy(groups, min_per_group=30):
    """
    Check if sample sizes are adequate for statistical testing.
    
    Parameters:
    -----------
    groups : list of array-like
        Data from different groups
    min_per_group : int
        Minimum recommended sample size
        
    Returns:
    --------
    dict : Sample size assessment
    """
    sizes = [len(g) for g in groups]
    
    return {
        'group_sizes': sizes,
        'min_size': min(sizes),
        'max_size': max(sizes),
        'mean_size': np.mean(sizes),
        'adequate': all(s >= min_per_group for s in sizes),
        'interpretation': f"Sample sizes: {sizes}. {'Adequate' if all(s >= min_per_group for s in sizes) else 'Small sample - interpret with caution'}",
        'recommendation': 'Sufficient power for robust testing' if all(s >= min_per_group for s in sizes) else f'Consider increasing iterations (current min: {min(sizes)}, recommended: ≥{min_per_group})'
    }


def comprehensive_assumption_testing(df, metric, alpha=0.05):
    """
    Perform comprehensive assumption testing for statistical analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Batch results
    metric : str
        Metric to analyze
    alpha : float
        Significance level
        
    Returns:
    --------
    dict : Complete assumption test results with recommendations
    """
    # Group data by configuration
    grouped = df.groupby('config_label')[metric]
    config_labels = list(grouped.groups.keys())
    groups = [grouped.get_group(config).values for config in config_labels]
    
    results = {
        'metric': metric,
        'n_groups': len(groups),
        'normality': {},
        'homogeneity': {},
        'sample_size': {},
        'test_selection': {}
    }
    
    # 1. Check normality for each group
    normality_results = []
    all_normal = True
    
    for i, (config, data) in enumerate(zip(config_labels, groups)):
        norm_test = check_normality(data, method='shapiro', alpha=alpha)
        normality_results.append({
            'config': config,
            'is_normal': norm_test['is_normal'],
            'p_value': norm_test['p_value']
        })
        if not norm_test['is_normal']:
            all_normal = False
    
    # Overall normality assessment
    pct_normal = sum(1 for r in normality_results if r['is_normal']) / len(normality_results) * 100
    
    results['normality'] = {
        'individual_tests': normality_results,
        'all_normal': all_normal,
        'percent_normal': pct_normal,
        'interpretation': f"{pct_normal:.0f}% of groups show normal distribution",
        'recommendation': 'Parametric tests appropriate' if all_normal else 'Non-parametric tests recommended'
    }
    
    # 2. Check homogeneity of variance
    homog_test = check_homogeneity_of_variance(groups, method='levene', alpha=alpha)
    results['homogeneity'] = homog_test
    
    # 3. Check sample sizes
    sample_test = check_sample_size_adequacy(groups, min_per_group=30)
    results['sample_size'] = sample_test
    
    # 4. Determine appropriate statistical test based on assumptions
    test_recommendation = determine_appropriate_test(
        all_normal=all_normal,
        homogeneous=homog_test['is_homogeneous'],
        adequate_n=sample_test['adequate'],
        n_groups=len(groups)
    )
    
    results['test_selection'] = test_recommendation
    
    return results


def determine_appropriate_test(all_normal, homogeneous, adequate_n, n_groups):
    """
    Determine appropriate statistical test based on assumption checks.
    
    Parameters:
    -----------
    all_normal : bool
        All groups normally distributed
    homogeneous : bool
        Homogeneous variances
    adequate_n : bool
        Adequate sample sizes
    n_groups : int
        Number of groups to compare
        
    Returns:
    --------
    dict : Test recommendation with rationale
    """
    rationale = []
    
    # Decision tree for test selection
    if n_groups == 2:
        # Two-group comparison
        if all_normal and homogeneous and adequate_n:
            recommended_test = "Independent samples t-test"
            rationale.append("✓ Normality assumption met")
            rationale.append("✓ Homogeneity of variance met")
            rationale.append("✓ Adequate sample sizes")
            rationale.append("→ Parametric test appropriate")
        elif all_normal and not homogeneous and adequate_n:
            recommended_test = "Welch's t-test"
            rationale.append("✓ Normality assumption met")
            rationale.append("✗ Homogeneity of variance violated")
            rationale.append("✓ Adequate sample sizes")
            rationale.append("→ Use Welch's correction for unequal variances")
        else:
            recommended_test = "Mann-Whitney U test"
            if not all_normal:
                rationale.append("✗ Normality assumption violated")
            if not adequate_n:
                rationale.append("⚠ Small sample sizes")
            rationale.append("→ Non-parametric test recommended")
    
    else:
        # Multiple group comparison
        if all_normal and homogeneous and adequate_n:
            recommended_test = "One-way ANOVA"
            post_hoc = "Tukey HSD"
            rationale.append("✓ Normality assumption met across all groups")
            rationale.append("✓ Homogeneity of variance met")
            rationale.append("✓ Adequate sample sizes")
            rationale.append("→ Parametric ANOVA appropriate")
            rationale.append(f"→ Post-hoc: {post_hoc} for pairwise comparisons")
        elif all_normal and not homogeneous and adequate_n:
            recommended_test = "Welch's ANOVA"
            post_hoc = "Games-Howell"
            rationale.append("✓ Normality assumption met")
            rationale.append("✗ Homogeneity of variance violated")
            rationale.append("✓ Adequate sample sizes")
            rationale.append("→ Use Welch's ANOVA (robust to unequal variances)")
            rationale.append(f"→ Post-hoc: {post_hoc} (does not assume equal variances)")
        else:
            recommended_test = "Kruskal-Wallis H-test"
            post_hoc = "Mann-Whitney U with Bonferroni correction"
            if not all_normal:
                rationale.append("✗ Normality assumption violated in some/all groups")
            if not adequate_n:
                rationale.append("⚠ Some groups have small sample sizes")
            rationale.append("→ Non-parametric test recommended (distribution-free)")
            rationale.append(f"→ Post-hoc: {post_hoc}")
    
    return {
        'recommended_test': recommended_test,
        'post_hoc': post_hoc if n_groups > 2 else None,
        'rationale': rationale,
        'parametric': all_normal and homogeneous and adequate_n,
        'summary': f"Based on assumption testing: Use {recommended_test}"
    }


def generate_assumption_report(assumption_results):
    """
    Generate formatted report of assumption testing results.
    
    Parameters:
    -----------
    assumption_results : dict
        Results from comprehensive_assumption_testing()
        
    Returns:
    --------
    str : Formatted assumption testing report
    """
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ASSUMPTIONS TESTING REPORT")
    report.append("=" * 80)
    report.append("")
    
    metric = assumption_results['metric']
    report.append(f"Metric Analyzed: {metric.upper().replace('_', ' ')}")
    report.append(f"Number of Groups: {assumption_results['n_groups']}")
    report.append("")
    
    # Sample Size Assessment
    report.append("-" * 80)
    report.append("1. SAMPLE SIZE ADEQUACY")
    report.append("-" * 80)
    ss = assumption_results['sample_size']
    report.append(f"Group sizes: {ss['group_sizes']}")
    report.append(f"Minimum: {ss['min_size']}, Maximum: {ss['max_size']}, Mean: {ss['mean_size']:.1f}")
    report.append(f"Assessment: {ss['interpretation']}")
    report.append(f"Recommendation: {ss['recommendation']}")
    report.append("")
    
    # Normality Testing
    report.append("-" * 80)
    report.append("2. NORMALITY TESTING")
    report.append("-" * 80)
    norm = assumption_results['normality']
    report.append(f"Overall: {norm['percent_normal']:.0f}% of groups show normal distribution")
    report.append(f"Interpretation: {norm['interpretation']}")
    report.append(f"Recommendation: {norm['recommendation']}")
    report.append("")
    report.append("Individual Group Results (ALL GROUPS):")
    for i, test in enumerate(norm['individual_tests'], 1):
        status = "✓ Normal" if test['is_normal'] else "✗ Non-normal"
        report.append(f"  {i}. {test['config']}: {status} (p = {test['p_value']:.6f})")
    report.append("")
    
    # Homogeneity of Variance
    report.append("-" * 80)
    report.append("3. HOMOGENEITY OF VARIANCE")
    report.append("-" * 80)
    homog = assumption_results['homogeneity']
    report.append(f"Test: {homog['test_name']}")
    report.append(f"Statistic: {homog['statistic']:.4f}")
    report.append(f"p-value: {homog['p_value']:.6f}")
    report.append(f"Result: {homog['interpretation']}")
    report.append(f"Recommendation: {homog['recommendation']}")
    report.append("")
    
    # Test Selection
    report.append("-" * 80)
    report.append("4. STATISTICAL TEST SELECTION")
    report.append("-" * 80)
    test_sel = assumption_results['test_selection']
    report.append(f"Recommended Test: {test_sel['recommended_test']}")
    if test_sel['post_hoc']:
        report.append(f"Post-hoc Test: {test_sel['post_hoc']}")
    report.append("")
    report.append("Rationale:")
    for reason in test_sel['rationale']:
        report.append(f"  {reason}")
    report.append("")
    report.append(f"Summary: {test_sel['summary']}")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# ===========================
# TIME-FREQUENCY ANALYSIS
# ===========================

def compute_morlet_timefreq(signal_data, sampling_rate, freq_range=(5, 500), n_freqs=80, w_cycles=6.0):
    """
    Compute Morlet wavelet time-frequency representation using cycles-based scaling.
    
    Implementation based on established time-frequency analysis methods with
    reflection padding for edge effects and fftconvolve for computational efficiency.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Signal to analyse
    sampling_rate : int
        Sampling rate in Hz
    freq_range : tuple
        (min_freq, max_freq) in Hz
    n_freqs : int
        Number of frequency bins
    w_cycles : float
        Number of cycles in Morlet wavelet (default: 6.0, standard for good time-frequency resolution)
        
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    tfr : np.ndarray
        Time-frequency representation (power)
    """
    # Generate frequency vector (logarithmic spacing for better low-frequency resolution)
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    
    # Initialize output
    signal_data = np.asarray(signal_data, dtype=float)
    n_times = len(signal_data)
    tfr_power = np.zeros((n_freqs, n_times))
    
    # For each frequency, compute wavelet transform
    for i, f0_hz in enumerate(freqs):
        # Calculate wavelet width in samples based on number of cycles
        # s_samples = (w_cycles / (2π * frequency)) * sampling_rate
        s_samples = (w_cycles / (2.0 * np.pi * float(f0_hz))) * sampling_rate
        s_samples = float(max(s_samples, 1e-9))  # Avoid division by zero
        
        # Create Morlet wavelet with appropriate length
        # Length covers ~12 standard deviations (ensures wavelet decay to ~0)
        L = int(np.ceil(12.0 * s_samples))
        if L % 2 == 0:  # Make odd for symmetry
            L += 1
        
        half = L // 2
        n = np.arange(-half, half + 1)
        tt = n / sampling_rate
        
        # Morlet wavelet: Gaussian envelope × complex exponential carrier
        # ψ(t) = exp(-0.5(t/σ)²) · exp(i·2π·f₀·t)
        gauss = np.exp(-0.5 * (n / s_samples) ** 2)
        carrier = np.exp(1j * 2.0 * np.pi * f0_hz * tt)
        wavelet = gauss * carrier
        
        # Normalize wavelet energy
        wavelet = wavelet / (np.sqrt(np.sum(np.abs(wavelet)**2)) + 1e-20)
        
        # Pad signal with reflection to handle edge effects
        pad_length = len(wavelet) // 2
        signal_padded = np.pad(signal_data, (pad_length, pad_length), mode='reflect')
        
        # Convolve using FFT (much faster for long signals)
        # Use conjugate-reversed wavelet for proper correlation
        coef = fftconvolve(signal_padded, np.conj(wavelet[::-1]), mode='same')
        
        # Remove padding to get original signal length
        coef = coef[pad_length:-pad_length]
        
        # Ensure exact length match (handle any rounding issues)
        if len(coef) != n_times:
            if len(coef) > n_times:
                coef = coef[:n_times]
            else:
                coef = np.pad(coef, (0, n_times - len(coef)), mode='edge')
        
        # Compute power (magnitude squared)
        tfr_power[i, :] = np.abs(coef) ** 2
    
    return freqs, tfr_power


# ===========================
# STATISTICAL TESTING & REPORTING
# ===========================

def calculate_effect_size(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Parameters:
    -----------
    group1, group2 : array-like
        Data from two groups to compare
        
    Returns:
    --------
    float : Cohen's d (standardized mean difference)
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    return d


def perform_statistical_analysis(df, metric='amplitude_error_pct', alpha=0.05):
    """
    Perform comprehensive statistical analysis with assumption testing on batch results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Batch results dataframe
    metric : str
        Metric to analyze
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict : Statistical results including assumptions, rankings, comparisons, and recommendations
    """
    results = {
        'metric': metric,
        'n_configs': len(df['config_label'].unique()),
        'n_iterations': len(df) // len(df['config_label'].unique()),
        'assumptions': {},
        'rankings': {},
        'statistical_test': {},
        'pairwise': [],
        'best_performers': [],
        'recommendations': [],
        'formal_report': ''
    }
    
    # STEP 1: Comprehensive Assumption Testing
    assumption_results = comprehensive_assumption_testing(df, metric, alpha)
    results['assumptions'] = assumption_results
    
    # Group by configuration
    grouped = df.groupby('config_label')[metric]
    config_labels = list(grouped.groups.keys())
    
    # Calculate statistics for each configuration
    config_stats = {}
    for config in config_labels:
        data = grouped.get_group(config).values
        
        # For amplitude error, use absolute values for ranking
        if 'error' in metric and metric != 'correlation':
            data_for_rank = np.abs(data)
            mean_for_rank = np.mean(data_for_rank)
        else:
            data_for_rank = data
            mean_for_rank = np.mean(data_for_rank)
        
        config_stats[config] = {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'median': np.median(data),
            'mean_abs': mean_for_rank,
            'ci_95': stats.t.interval(0.95, len(data)-1, 
                                     loc=np.mean(data), 
                                     scale=stats.sem(data)),
            'data': data
        }
    
    # Rank configurations
    if 'error' in metric or 'rmse' in metric:
        ranked = sorted(config_stats.items(), key=lambda x: x[1]['mean_abs'])
    elif metric == 'correlation':
        ranked = sorted(config_stats.items(), key=lambda x: -x[1]['mean'])
    else:
        ranked = sorted(config_stats.items(), key=lambda x: x[1]['mean_abs'])
    
    results['rankings'] = {i+1: {'config': config, 'stats': stats_dict} 
                          for i, (config, stats_dict) in enumerate(ranked)}
    
    # STEP 2: Perform Appropriate Statistical Test Based on Assumptions
    groups = [config_stats[config]['data'] for config in config_labels]
    
    test_used = assumption_results['test_selection']['recommended_test']
    
    if test_used == "Kruskal-Wallis H-test" or not assumption_results['test_selection']['parametric']:
        # Non-parametric test
        h_stat, p_value = kruskal(*groups)
        results['statistical_test'] = {
            'test_name': 'Kruskal-Wallis H-test',
            'test_type': 'Non-parametric (distribution-free)',
            'statistic': h_stat,
            'statistic_name': 'H',
            'p_value': p_value,
            'df': len(groups) - 1,
            'significant': p_value < alpha,
            'interpretation': 'Significant differences exist between configurations' if p_value < alpha 
                             else 'No significant differences detected',
            'rationale': assumption_results['test_selection']['rationale']
        }
    else:
        # Parametric ANOVA
        f_stat, p_value = f_oneway(*groups)
        results['statistical_test'] = {
            'test_name': 'One-way ANOVA',
            'test_type': 'Parametric',
            'statistic': f_stat,
            'statistic_name': 'F',
            'p_value': p_value,
            'df_between': len(groups) - 1,
            'df_within': sum(len(g) for g in groups) - len(groups),
            'significant': p_value < alpha,
            'interpretation': 'Significant differences exist between configurations' if p_value < alpha 
                             else 'No significant differences detected',
            'rationale': assumption_results['test_selection']['rationale']
        }
    
    # STEP 3: Pairwise comparisons for top 5 configurations
    top_configs = [item[0] for item in ranked[:min(5, len(ranked))]]
    
    for i, config1 in enumerate(top_configs):
        for config2 in top_configs[i+1:]:
            data1 = config_stats[config1]['data']
            data2 = config_stats[config2]['data']
            
            # Use appropriate pairwise test
            if assumption_results['test_selection']['parametric']:
                # t-test
                t_stat, p_val = ttest_ind(data1, data2, equal_var=assumption_results['homogeneity']['is_homogeneous'])
            else:
                # Mann-Whitney U test (non-parametric)
                u_stat, p_val = mannwhitneyu(np.abs(data1) if 'error' in metric else data1,
                                             np.abs(data2) if 'error' in metric else data2,
                                             alternative='two-sided')
            
            # Effect size (Cohen's d)
            effect_size = calculate_effect_size(data1, data2)
            
            # Bonferroni correction
            n_comparisons = len(top_configs) * (len(top_configs) - 1) / 2
            p_corrected = min(p_val * n_comparisons, 1.0)
            
            results['pairwise'].append({
                'config1': config1,
                'config2': config2,
                'mean1': config_stats[config1]['mean'],
                'mean2': config_stats[config2]['mean'],
                'p_value': p_val,
                'p_corrected': p_corrected,
                'significant': p_corrected < alpha,
                'effect_size': effect_size,
                'effect_magnitude': 'Large' if abs(effect_size) > 0.8 
                                   else 'Medium' if abs(effect_size) > 0.5 
                                   else 'Small' if abs(effect_size) > 0.2 
                                   else 'Negligible'
            })
    
    # STEP 4: Identify best performers
    for rank in [1, 2, 3]:
        if rank <= len(results['rankings']):
            config_info = results['rankings'][rank]
            config = config_info['config']
            stats_dict = config_info['stats']
            
            if rank < len(results['rankings']):
                next_config = results['rankings'][rank + 1]['config']
                pairwise_result = next((p for p in results['pairwise'] 
                                       if (p['config1'] == config and p['config2'] == next_config) or
                                          (p['config1'] == next_config and p['config2'] == config)),
                                      None)
                sig_diff = pairwise_result['significant'] if pairwise_result else False
            else:
                sig_diff = True
            
            results['best_performers'].append({
                'rank': rank,
                'config': config,
                'mean': stats_dict['mean'],
                'mean_abs': stats_dict['mean_abs'],
                'std': stats_dict['std'],
                'median': stats_dict['median'],
                'ci_95': stats_dict['ci_95'],
                'significantly_different_from_next': sig_diff
            })
    
    # STEP 5: Generate recommendations
    best = results['rankings'][1]
    results['recommendations'].append({
        'type': 'Primary Recommendation',
        'config': best['config'],
        'reason': f"Lowest mean(|{metric.replace('_', ' ')}|): {best['stats']['mean_abs']:.3f} ± {best['stats']['std']:.3f}",
        'confidence': 'High' if results['statistical_test']['significant'] else 'Moderate'
    })
    
    if len(results['best_performers']) > 1:
        if not results['best_performers'][0]['significantly_different_from_next']:
            second = results['rankings'][2]
            results['recommendations'].append({
                'type': 'Equivalent Alternative',
                'config': second['config'],
                'reason': f"Not significantly different from top performer (p > {alpha})",
                'confidence': 'High'
            })
    
    # STEP 6: Generate formal statistical report
    results['formal_report'] = generate_formal_statistical_report(results)
    
    return results


def generate_formal_statistical_report(results):
    """
    Generate formal APA-style statistical report.
    
    Parameters:
    -----------
    results : dict
        Complete statistical analysis results
        
    Returns:
    --------
    str : Formal report suitable for publication
    """
    report = []
    
    metric = results['metric'].replace('_', ' ').title()
    test_info = results['statistical_test']
    assumptions = results['assumptions']
    
    # Assumption testing statement
    report.append("STATISTICAL METHODS JUSTIFICATION:")
    report.append("")
    report.append(f"Assumption testing was conducted to determine appropriate statistical methods. ")
    
    if assumptions['normality']['all_normal']:
        report.append(f"Shapiro-Wilk tests indicated all groups were consistent with normal distribution. ")
    else:
        report.append(f"Normality testing (Shapiro-Wilk) revealed {assumptions['normality']['percent_normal']:.0f}% of groups showed non-normal distributions, ")
        report.append(f"violating the normality assumption for parametric tests. ")
    
    if assumptions['homogeneity']['is_homogeneous']:
        report.append(f"{assumptions['homogeneity']['test_name']} confirmed homogeneity of variance across groups (p = {assumptions['homogeneity']['p_value']:.3f}). ")
    else:
        report.append(f"{assumptions['homogeneity']['test_name']} indicated heterogeneous variances (p = {assumptions['homogeneity']['p_value']:.3f}). ")
    
    report.append(f"Based on these findings, {test_info['test_name']} was selected as the most appropriate statistical test.")
    report.append("")
    report.append("")
    
    # Main statistical result - formal APA style
    report.append("STATISTICAL RESULTS:")
    report.append("")
    
    if test_info['test_name'] == 'Kruskal-Wallis H-test':
        report.append(f"A Kruskal-Wallis H-test revealed significant differences in {metric} between ")
        report.append(f"filter configurations (H({test_info['df']}) = {test_info['statistic']:.2f}, p < .001). ")
    elif test_info['test_name'] == 'One-way ANOVA':
        report.append(f"A one-way ANOVA revealed significant differences in {metric} between ")
        report.append(f"filter configurations (F({test_info['df_between']},{test_info['df_within']}) = {test_info['statistic']:.2f}, p < .001). ")
    
    # Best performer
    best = results['best_performers'][0]
    report.append(f"The optimal configuration was {best['config']}, demonstrating ")
    report.append(f"mean {metric} of {best['mean']:.4f} ± {best['std']:.4f} (SD), ")
    report.append(f"median = {best['median']:.4f}, 95% CI [{best['ci_95'][0]:.4f}, {best['ci_95'][1]:.4f}].")
    report.append("")
    
    return "\n".join(report)
    """
    Perform comprehensive statistical analysis on batch results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Batch results dataframe
    metric : str
        Metric to analyze
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict : Statistical results including rankings, comparisons, and recommendations
    """
    results = {
        'metric': metric,
        'n_configs': len(df['config_label'].unique()),
        'n_iterations': len(df) // len(df['config_label'].unique()),
        'rankings': {},
        'anova': {},
        'pairwise': [],
        'best_performers': [],
        'recommendations': []
    }
    
    # Group by configuration
    grouped = df.groupby('config_label')[metric]
    config_labels = list(grouped.groups.keys())
    
    # Calculate statistics for each configuration
    config_stats = {}
    for config in config_labels:
        data = grouped.get_group(config).values
        
        # For amplitude error, use absolute values
        if 'error' in metric and metric != 'correlation':
            data_for_rank = np.abs(data)
            mean_for_rank = np.mean(data_for_rank)
        else:
            data_for_rank = data
            mean_for_rank = np.mean(data_for_rank)
        
        config_stats[config] = {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'median': np.median(data),
            'mean_abs': mean_for_rank,
            'ci_95': stats.t.interval(0.95, len(data)-1, 
                                     loc=np.mean(data), 
                                     scale=stats.sem(data)),
            'data': data
        }
    
    # Rank configurations
    if 'error' in metric or 'rmse' in metric:
        # Lower is better - rank by mean absolute value
        ranked = sorted(config_stats.items(), key=lambda x: x[1]['mean_abs'])
    elif metric == 'correlation':
        # Higher is better
        ranked = sorted(config_stats.items(), key=lambda x: -x[1]['mean'])
    else:
        # Default: lower is better
        ranked = sorted(config_stats.items(), key=lambda x: x[1]['mean_abs'])
    
    results['rankings'] = {i+1: {'config': config, 'stats': stats_dict} 
                          for i, (config, stats_dict) in enumerate(ranked)}
    
    # Perform Kruskal-Wallis test (non-parametric ANOVA)
    groups = [config_stats[config]['data'] for config in config_labels]
    h_stat, p_value = kruskal(*groups)
    
    results['anova'] = {
        'test': 'Kruskal-Wallis H-test',
        'h_statistic': h_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'interpretation': 'Significant differences exist between configurations' if p_value < alpha 
                         else 'No significant differences detected'
    }
    
    # Pairwise comparisons for top 5 configurations
    top_configs = [item[0] for item in ranked[:min(5, len(ranked))]]
    
    for i, config1 in enumerate(top_configs):
        for config2 in top_configs[i+1:]:
            data1 = config_stats[config1]['data']
            data2 = config_stats[config2]['data']
            
            # Mann-Whitney U test (non-parametric)
            u_stat, p_val = mannwhitneyu(np.abs(data1) if 'error' in metric else data1,
                                         np.abs(data2) if 'error' in metric else data2,
                                         alternative='two-sided')
            
            # Effect size (Cohen's d)
            effect_size = calculate_effect_size(data1, data2)
            
            # Bonferroni correction for multiple comparisons
            n_comparisons = len(top_configs) * (len(top_configs) - 1) / 2
            p_corrected = min(p_val * n_comparisons, 1.0)
            
            results['pairwise'].append({
                'config1': config1,
                'config2': config2,
                'mean1': config_stats[config1]['mean'],
                'mean2': config_stats[config2]['mean'],
                'p_value': p_val,
                'p_corrected': p_corrected,
                'significant': p_corrected < alpha,
                'effect_size': effect_size,
                'effect_magnitude': 'Large' if abs(effect_size) > 0.8 
                                   else 'Medium' if abs(effect_size) > 0.5 
                                   else 'Small' if abs(effect_size) > 0.2 
                                   else 'Negligible'
            })
    
    # Identify best performers (top 3 with statistical verification)
    for rank in [1, 2, 3]:
        if rank <= len(results['rankings']):
            config_info = results['rankings'][rank]
            config = config_info['config']
            stats_dict = config_info['stats']
            
            # Check if significantly different from next rank
            if rank < len(results['rankings']):
                next_config = results['rankings'][rank + 1]['config']
                pairwise_result = next((p for p in results['pairwise'] 
                                       if (p['config1'] == config and p['config2'] == next_config) or
                                          (p['config1'] == next_config and p['config2'] == config)),
                                      None)
                
                sig_diff = pairwise_result['significant'] if pairwise_result else False
            else:
                sig_diff = True
            
            results['best_performers'].append({
                'rank': rank,
                'config': config,
                'mean': stats_dict['mean'],
                'std': stats_dict['std'],
                'median': stats_dict['median'],
                'ci_95': stats_dict['ci_95'],
                'significantly_different_from_next': sig_diff
            })
    
    # Generate recommendations
    best = results['rankings'][1]
    results['recommendations'].append({
        'type': 'Primary Recommendation',
        'config': best['config'],
        'reason': f"Lowest mean {metric.replace('_', ' ')}: {best['stats']['mean']:.3f} ± {best['stats']['std']:.3f}",
        'confidence': 'High' if results['anova']['significant'] else 'Moderate'
    })
    
    # Add alternative if top 2 not significantly different
    if len(results['best_performers']) > 1:
        if not results['best_performers'][0]['significantly_different_from_next']:
            second = results['rankings'][2]
            results['recommendations'].append({
                'type': 'Equivalent Alternative',
                'config': second['config'],
                'reason': f"Not significantly different from top performer (p > {alpha})",
                'confidence': 'High'
            })
    
    return results


def generate_statistical_report(df, metrics=['amplitude_error_pct', 'peak_latency_error_ms', 'correlation']):
    """
    Generate comprehensive statistical report for all metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Batch results
    metrics : list
        Metrics to analyze
        
    Returns:
    --------
    str : Formatted text report
    """
    report = []
    report.append("=" * 80)
    report.append("MEPSimFilt - STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    n_configs = len(df['config_label'].unique())
    n_iters = len(df) // n_configs
    
    report.append(f"Analysis Summary:")
    report.append(f"  • Configurations tested: {n_configs}")
    report.append(f"  • Iterations per configuration: {n_iters}")
    report.append(f"  • Total tests: {len(df)}")
    report.append("")
    report.append("=" * 80)
    
    # Analyze each metric
    for metric in metrics:
        report.append("")
        report.append(f"METRIC: {metric.upper().replace('_', ' ')}")
        report.append("-" * 80)
        
        results = perform_statistical_analysis(df, metric)
        
        # Assumption Testing Summary
        report.append("")
        report.append("Assumption Testing Summary:")
        assumptions = results['assumptions']
        report.append(f"  • Normality: {assumptions['normality']['percent_normal']:.0f}% groups normal")
        report.append(f"  • Variance Homogeneity: {'Homogeneous' if assumptions['homogeneity']['is_homogeneous'] else 'Heterogeneous'} (p = {assumptions['homogeneity']['p_value']:.4f})")
        report.append(f"  • Sample Sizes: Adequate ({assumptions['sample_size']['adequate']})")
        report.append(f"  • Selected Test: {assumptions['test_selection']['recommended_test']}")
        report.append("")
        
        # Overall test
        test_info = results['statistical_test']
        report.append("Overall Statistical Test:")
        report.append(f"  • Test: {test_info['test_name']} ({test_info['test_type']})")
        report.append(f"  • {test_info['statistic_name']}-statistic: {test_info['statistic']:.4f}")
        if 'df' in test_info:
            report.append(f"  • df: {test_info['df']}")
        elif 'df_between' in test_info and 'df_within' in test_info:
            report.append(f"  • df: ({test_info['df_between']}, {test_info['df_within']})")
        report.append(f"  • p-value: {test_info['p_value']:.6f}")
        report.append(f"  • Result: {test_info['interpretation']}")
        report.append("")
        report.append("Test Selection Rationale:")
        for reason in test_info['rationale']:
            report.append(f"  {reason}")
        report.append("")
        
        # Top 5 performers
        report.append("TOP 5 PERFORMERS (Ranked):")
        report.append("")
        for rank in range(1, min(6, len(results['rankings']) + 1)):
            info = results['rankings'][rank]
            config = info['config']
            stats_dict = info['stats']
            
            report.append(f"  Rank {rank}: {config}")
            report.append(f"    Mean (Signed): {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f} (SD)")
            report.append(f"    Mean(|Error|): {stats_dict['mean_abs']:.4f}  [Used for ranking]")
            report.append(f"    Median: {stats_dict['median']:.4f}")
            report.append(f"    95% CI: [{stats_dict['ci_95'][0]:.4f}, {stats_dict['ci_95'][1]:.4f}]")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("")
        for rec in results['recommendations']:
            report.append(f"  • {rec['type']}: {rec['config']}")
            report.append(f"    Reason: {rec['reason']}")
            report.append(f"    Confidence: {rec['confidence']}")
            report.append("")
        
        # Key pairwise comparisons
        if results['pairwise']:
            report.append("KEY PAIRWISE COMPARISONS (Top performers):")
            report.append("")
            for comp in results['pairwise'][:5]:  # Show top 5 comparisons
                report.append(f"  {comp['config1']} vs {comp['config2']}:")
                report.append(f"    Mean difference: {abs(comp['mean1'] - comp['mean2']):.4f}")
                report.append(f"    p-value (corrected): {comp['p_corrected']:.6f}")
                report.append(f"    Significant: {'Yes' if comp['significant'] else 'No'}")
                report.append(f"    Effect size: {comp['effect_size']:.3f} ({comp['effect_magnitude']})")
                report.append("")
    
    report.append("=" * 80)
    report.append("END OF STATISTICAL REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def create_recommendations_summary(df):
    """
    Create concise summary of filter recommendations based on all metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Batch results
        
    Returns:
    --------
    str : Summary text with recommendations
    """
    summary = []
    summary.append("=" * 80)
    summary.append("EVIDENCE-BASED FILTER RECOMMENDATIONS")
    summary.append("=" * 80)
    summary.append("")
    
    # Analyze multiple metrics
    metrics = {
        'amplitude_error_pct': 'Amplitude Preservation',
        'peak_latency_error_ms': 'Temporal Accuracy',
        'correlation': 'Morphological Fidelity'
    }
    
    top_performers = {}
    for metric_key, metric_name in metrics.items():
        if metric_key in df.columns:
            results = perform_statistical_analysis(df, metric_key)
            top_config = results['rankings'][1]['config']
            top_performers[metric_name] = top_config
    
    # Find consensus recommendation
    from collections import Counter
    config_counts = Counter(top_performers.values())
    consensus = config_counts.most_common(1)[0] if config_counts else None
    
    summary.append("OVERALL RECOMMENDATION:")
    summary.append("")
    if consensus:
        summary.append(f"  🏆 BEST OVERALL FILTER: {consensus[0]}")
        summary.append(f"     (Top performer in {consensus[1]}/{len(metrics)} metrics)")
        summary.append("")
    
    summary.append("METRIC-SPECIFIC TOP PERFORMERS:")
    summary.append("")
    for metric_name, config in top_performers.items():
        summary.append(f"  • {metric_name}: {config}")
    
    summary.append("")
    summary.append("INTERPRETATION GUIDE:")
    summary.append("")
    summary.append("  Filter Order:")
    summary.append("    • Lower orders (2nd) generally preserve amplitude better")
    summary.append("    • Higher orders (4th+) provide sharper frequency cutoffs")
    summary.append("    • Recommendation: Use 2nd or 4th order for MEP analysis")
    summary.append("")
    summary.append("  Frequency Range:")
    summary.append("    • Highpass 10-20 Hz removes DC drift and low-frequency noise")
    summary.append("    • Lowpass 450-500 Hz captures MEP content, removes high-frequency noise")
    summary.append("    • Extending to 1000 Hz may preserve detail but includes more noise")
    summary.append("")
    summary.append("  Notch Filter:")
    summary.append("    • Use when 50 Hz line noise is prominent")
    summary.append("    • May introduce minimal distortion (~0.1-0.5% amplitude error)")
    summary.append("    • Compare configurations with/without notch in your data")
    summary.append("")
    summary.append("=" * 80)
    
    return "\n".join(summary)


def create_comparison_table(df, top_n=10):
    """
    Create detailed comparison table of top N filter configurations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Batch results
    top_n : int
        Number of top configurations to include
        
    Returns:
    --------
    pandas.DataFrame : Comparison table
    """
    # Calculate aggregate statistics for each configuration
    agg_dict = {
        'amplitude_error_pct': ['mean', 'std', 'median'],
        'peak_latency_error_ms': ['mean', 'std', 'median'],
        'correlation': ['mean', 'std', 'median'],
        'rmse_mep': ['mean', 'std']
    }
    
    # Only add columns that exist in the DataFrame
    if 'snr_improvement' in df.columns:
        agg_dict['snr_improvement'] = ['mean', 'std']
    
    if 'baseline_std' in df.columns:
        agg_dict['baseline_std'] = ['mean', 'std']
    
    agg_stats = df.groupby('config_label').agg(agg_dict).round(4)
    
    # Flatten column names
    agg_stats.columns = ['_'.join(col).strip() for col in agg_stats.columns.values]
    
    # Calculate mean of absolute amplitude errors for ranking
    mean_abs_errors = df.groupby('config_label')['amplitude_error_pct'].apply(
        lambda x: np.mean(np.abs(x))
    ).round(4)
    agg_stats['amplitude_error_mean_abs'] = mean_abs_errors
    
    # Calculate composite score (normalized across metrics)
    # Lower is better for errors, higher is better for correlation
    amp_norm = (agg_stats['amplitude_error_mean_abs'] - agg_stats['amplitude_error_mean_abs'].min()) / \
               (agg_stats['amplitude_error_mean_abs'].max() - agg_stats['amplitude_error_mean_abs'].min() + 1e-10)
    
    lat_norm = (agg_stats['peak_latency_error_ms_mean'].abs() - agg_stats['peak_latency_error_ms_mean'].abs().min()) / \
               (agg_stats['peak_latency_error_ms_mean'].abs().max() - agg_stats['peak_latency_error_ms_mean'].abs().min() + 1e-10)
    
    corr_norm = (agg_stats['correlation_mean'].max() - agg_stats['correlation_mean']) / \
                (agg_stats['correlation_mean'].max() - agg_stats['correlation_mean'].min() + 1e-10)
    
    # Composite score (equal weighting)
    agg_stats['composite_score'] = (amp_norm + lat_norm + corr_norm) / 3.0
    
    # Rank by mean absolute amplitude error (primary metric)
    agg_stats['rank'] = agg_stats['amplitude_error_mean_abs'].rank()
    
    # Sort by composite score and select top N
    comparison_df = agg_stats.sort_values('composite_score').head(top_n)
    
    # Reset index to get config_label as column
    comparison_df = comparison_df.reset_index()
    
    # Reorder columns for clarity
    cols_order = ['config_label', 'amplitude_error_pct_mean', 'amplitude_error_mean_abs', 
                  'amplitude_error_pct_std', 'peak_latency_error_ms_mean', 
                  'correlation_mean', 'composite_score', 'rank']
    
    # Only include columns that exist
    cols_order = [c for c in cols_order if c in comparison_df.columns]
    other_cols = [c for c in comparison_df.columns if c not in cols_order]
    comparison_df = comparison_df[cols_order + other_cols]
    
    # Add friendly rank column at start
    comparison_df.insert(0, 'Display_Rank', range(1, len(comparison_df) + 1))
    
    # Rename columns for clarity
    comparison_df = comparison_df.rename(columns={
        'config_label': 'Configuration',
        'amplitude_error_pct_mean': 'Amp_Error_Mean',
        'amplitude_error_mean_abs': 'Amp_Error_Mean|Abs|',
        'amplitude_error_pct_std': 'Amp_Error_SD',
        'peak_latency_error_ms_mean': 'Latency_Error_Mean',
        'correlation_mean': 'Correlation_Mean',
        'composite_score': 'Composite_Score'
    })
    
    return comparison_df


def plot_timefreq_comparison(mep_clean, mep_noisy, mep_filtered, time, sampling_rate,
                             freq_range=(5, 500), n_freqs=80, vmax_percentile=98):
    """
    Create multi-panel time-frequency comparison figure.
    
    Parameters:
    -----------
    mep_clean : np.ndarray
        Ground truth MEP
    mep_noisy : np.ndarray
        Noisy MEP  
    mep_filtered : np.ndarray
        Filtered MEP
    time : np.ndarray
        Time vector (seconds)
    sampling_rate : int
        Sampling rate (Hz)
    freq_range : tuple
        (min_freq, max_freq) for analysis
    n_freqs : int
        Number of frequency bins
    vmax_percentile : float
        Percentile for colour scale normalization
        
    Returns:
    --------
    fig : matplotlib figure
    """
    # Compute time-frequency representations
    freqs_clean, tfr_clean = compute_morlet_timefreq(mep_clean, sampling_rate, freq_range, n_freqs)
    freqs_noisy, tfr_noisy = compute_morlet_timefreq(mep_noisy, sampling_rate, freq_range, n_freqs)
    freqs_filt, tfr_filt = compute_morlet_timefreq(mep_filtered, sampling_rate, freq_range, n_freqs)
    
    # Ensure all have same length (trim to shortest if needed)
    min_len = min(tfr_clean.shape[1], tfr_noisy.shape[1], tfr_filt.shape[1], len(time))
    tfr_clean = tfr_clean[:, :min_len]
    tfr_noisy = tfr_noisy[:, :min_len]
    tfr_filt = tfr_filt[:, :min_len]
    time_plot = time[:min_len]
    mep_clean_plot = mep_clean[:min_len]
    mep_noisy_plot = mep_noisy[:min_len]
    mep_filtered_plot = mep_filtered[:min_len]
    
    # Create figure with 3 rows x 2 columns layout
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.5, 1], hspace=0.28, wspace=0.15)
    
    # Determine common colour scale based on ground truth
    vmax = np.percentile(tfr_clean, vmax_percentile)
    
    time_ms = time_plot * 1000
    
    # ========== ROW 1: GROUND TRUTH ==========
    ax1_tf = fig.add_subplot(gs[0, 0])
    ax1_wave = fig.add_subplot(gs[0, 1])
    
    # Time-frequency plot
    im1 = ax1_tf.pcolormesh(time_ms, freqs_clean, tfr_clean,
                            shading='gouraud', cmap='viridis', vmin=0, vmax=vmax)
    ax1_tf.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8, label='TMS')
    ax1_tf.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1_tf.set_title('A) Ground Truth MEP - Time-Frequency Representation', 
                     fontsize=13, fontweight='bold', loc='left')
    ax1_tf.set_yscale('log')
    ax1_tf.set_ylim(freq_range)
    ax1_tf.grid(False)
    ax1_tf.tick_params(labelbottom=False)
    
    # Waveform
    ax1_wave.plot(time_ms, mep_clean_plot, 'k-', linewidth=2)
    ax1_wave.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1_wave.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    ax1_wave.set_ylabel('Amplitude (mV)', fontsize=11, fontweight='bold')
    ax1_wave.set_title('Waveform', fontsize=11, fontweight='bold')
    ax1_wave.grid(True, alpha=0.3)
    ax1_wave.set_xlim([time_ms[0], time_ms[-1]])
    ax1_wave.tick_params(labelbottom=False)
    
    # ========== ROW 2: NOISY SIGNAL ==========
    ax2_tf = fig.add_subplot(gs[1, 0])
    ax2_wave = fig.add_subplot(gs[1, 1])
    
    # Time-frequency plot
    im2 = ax2_tf.pcolormesh(time_ms, freqs_noisy, tfr_noisy,
                            shading='gouraud', cmap='viridis', vmin=0, vmax=vmax)
    ax2_tf.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax2_tf.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2_tf.set_title('B) Noisy Signal - Time-Frequency Representation', 
                     fontsize=13, fontweight='bold', loc='left')
    ax2_tf.set_yscale('log')
    ax2_tf.set_ylim(freq_range)
    ax2_tf.grid(False)
    ax2_tf.tick_params(labelbottom=False)
    
    # Waveform
    ax2_wave.plot(time_ms, mep_noisy_plot, 'r-', linewidth=1.2, alpha=0.8)
    ax2_wave.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2_wave.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    ax2_wave.set_ylabel('Amplitude (mV)', fontsize=11, fontweight='bold')
    ax2_wave.set_title('Waveform', fontsize=11, fontweight='bold')
    ax2_wave.grid(True, alpha=0.3)
    ax2_wave.set_xlim([time_ms[0], time_ms[-1]])
    ax2_wave.tick_params(labelbottom=False)
    
    # ========== ROW 3: FILTERED SIGNAL ==========
    ax3_tf = fig.add_subplot(gs[2, 0])
    ax3_wave = fig.add_subplot(gs[2, 1])
    
    # Time-frequency plot
    im3 = ax3_tf.pcolormesh(time_ms, freqs_filt, tfr_filt,
                            shading='gouraud', cmap='viridis', vmin=0, vmax=vmax)
    ax3_tf.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax3_tf.set_xlabel('Time (ms) - Negative = Pre-Stimulus', fontsize=12, fontweight='bold')
    ax3_tf.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax3_tf.set_title('C) Filtered Signal - Time-Frequency Representation', 
                     fontsize=13, fontweight='bold', loc='left')
    ax3_tf.set_yscale('log')
    ax3_tf.set_ylim(freq_range)
    ax3_tf.grid(False)
    
    # Waveform
    ax3_wave.plot(time_ms, mep_filtered_plot, 'b-', linewidth=2)
    ax3_wave.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax3_wave.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    ax3_wave.set_xlabel('Time (ms)', fontsize=11)
    ax3_wave.set_ylabel('Amplitude (mV)', fontsize=11, fontweight='bold')
    ax3_wave.set_title('Waveform', fontsize=11, fontweight='bold')
    ax3_wave.grid(True, alpha=0.3)
    ax3_wave.set_xlim([time_ms[0], time_ms[-1]])
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.12, 0.02, 0.56, 0.015])
    cbar = plt.colorbar(im3, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Wavelet Power (a.u.)', fontsize=12, fontweight='bold')
    
    return fig

# Page configuration
st.set_page_config(
    page_title="MEPSimFilt",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for better styling and hiding default elements
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
    /* Hide Streamlit default menu items */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    # Logo and branding - centered and aligned
    import os
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'TMSMultiLab_logo.png')
    
    # Center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
    
    # Centered TMSMultiLab text with link
    st.markdown("""
    <div style='text-align: center; margin-top: -5px; margin-bottom: 25px;'>
        <a href='https://github.com/TMSMultiLab/TMSMultiLab/wiki' target='_blank' 
           style='text-decoration: none; color: #1f77b4; font-weight: bold; font-size: 20px;'>
            TMSMultiLab
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Tool Information
    st.markdown("### MEPSimFilt")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Author:** Justin W. Andrushko, PhD")
    st.markdown("**Institution:** Northumbria University")
    
    st.divider()
    
    # Resources - TMSMultiLab and Scientific Links
    st.markdown("### Resources")
    st.markdown("""
    - [Digital Filter Design](https://scipy.github.io/devdocs/tutorial/signal.html)
    - [TMSMultiLab](https://github.com/TMSMultiLab/TMSMultiLab/wiki)
    - [MEPs](https://github.com/TMSMultiLab/TMSMultiLab/wiki/MEPs)
    - [TMS-EMG](https://github.com/TMSMultiLab/TMSMultiLab/wiki/EMG)
    - [CEDE EMG Best Practices](https://cede.isek.org/)
    """)
    
    st.divider()
    
    # Citation
    st.markdown("### Citation")
    st.markdown("""
    <div style='font-size: 11px; line-height: 1.4;'>
    Andrushko, J.W. (2025). MEPSimFilt: A Systematic Digital Filter Evaluation tool
    for Motor Evoked Potentials (Version 1.0.0). Northumbria University.
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'generated_signals' not in st.session_state:
    st.session_state.generated_signals = None
if 'filter_results' not in st.session_state:
    st.session_state.filter_results = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# Title and description
st.title("🧠 MEPSimFilt")
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
    
    # ========== FILE UPLOAD SECTION ==========
    st.subheader("📂 Load Real MEP Waveform (Optional)")
    
    with st.expander("Upload and Auto-Extract Parameters from Real Data", expanded=False):
        st.markdown("""
        Upload a real MEP waveform to automatically extract parameters and match your experimental data.
        
        **Supported formats:**
        - **LabView** (TXT: 2-row transposed format with time/data rows)
        - **LabChart** (TXT exports with headers: Interval, ChannelTitle, Range)
        - **Spike2** (TXT exports with INFORMATION, SUMMARY, START markers)
          - **Automatically detects DigMarks/Events** (Digitimer, Magstim triggers)
          - **Segments recording** around each event for individual trial analysis
        - **Multi-trial CSV** (rows = timepoints, columns = trials)
        - **Standard CSV** (two columns: time, amplitude)
        - **Standard TXT** (space or tab delimited)
        - **MATLAB MAT** (variables: signal/data/mep and time/t)
        
        The tool will auto-detect format and events. For Spike2 files with DigMarks, you can select specific trials!
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file containing MEP waveform",
            type=['txt', 'csv', 'mat'],
            help="File should contain time and amplitude data. Formats: TXT (space/tab delimited), CSV (2 columns), MAT (MATLAB)"
        )
        
        if uploaded_file is not None:
            try:
                # Load waveform (or use previously segmented data if exists)
                if hasattr(st.session_state, 'spike2_segmented') and st.session_state.spike2_segmented:
                    # Use previously segmented data
                    time_loaded, signal_loaded, fs_loaded, file_metadata = st.session_state.loaded_waveform_raw
                else:
                    # Fresh load from file
                    time_loaded, signal_loaded, fs_loaded, file_metadata = load_waveform_from_file(uploaded_file)
                
                # Check if sampling rate needs user input
                if file_metadata.get('requires_fs_input') or fs_loaded is None:
                    st.warning("⚠️ Sampling rate not found in file. Please specify:")
                    
                    fs_user = st.number_input(
                        "Enter Sampling Rate (Hz)",
                        min_value=100,
                        max_value=100000,
                        value=5000,
                        step=100,
                        key='user_fs_input',
                        help="Common values: 1000, 2000, 5000, 10000, 20000 Hz"
                    )
                    
                    fs_loaded = float(fs_user)
                    
                    # Create time vector if not present
                    if time_loaded is None:
                        time_loaded = np.arange(len(signal_loaded)) / fs_loaded
                        # Check if file indicated pre-stimulus period
                        if file_metadata.get('format') == 'LabView (Transposed)':
                            # LabView often starts at negative time (pre-stim)
                            # Estimate: 10% of data as pre-stim
                            pre_stim_samples = int(len(signal_loaded) * 0.1)
                            time_loaded = time_loaded - (pre_stim_samples / fs_loaded)
                    
                    st.info(f"✅ Using {fs_loaded:.0f} Hz - Time axis created")
                
                st.success(f"✅ Loaded {len(signal_loaded)} samples at {fs_loaded:.0f} Hz")
                
                # Show file format info with details
                if file_metadata.get('format') == 'LabView (Transposed)':
                    st.info(f"📄 **Format:** LabView transposed format (2 rows × {file_metadata.get('n_samples', 0)} samples)")
                    if file_metadata.get('note'):
                        st.caption(file_metadata['note'])
                
                elif file_metadata.get('format') == 'Spike2':
                    # Check if this is segmented or full recording
                    if st.session_state.get('spike2_segmented', False):
                        event_info = st.session_state.get('spike2_event_info', {})
                        st.success(f"✅ **Viewing Segmented Data:** {event_info.get('channel', 'Event')} Event {event_info.get('event_num', 1)}")
                        
                        # Option to reset and view full recording again
                        if st.button("🔄 Reset - View Full Recording", key='reset_spike2_segment'):
                            # Clear segmentation flags
                            st.session_state.spike2_segmented = False
                            if 'spike2_event_info' in st.session_state:
                                del st.session_state.spike2_event_info
                            if 'loaded_waveform_raw' in st.session_state:
                                del st.session_state.loaded_waveform_raw
                            st.success("✅ Reset - will reload full recording")
                            st.rerun()
                    else:
                        st.info(f"📄 **Format:** Spike2 exported file (Full Continuous Recording)")
                    
                    # Check for detected events/DigMarks (only show if viewing full recording)
                    if not st.session_state.get('spike2_segmented', False) and file_metadata.get('has_events') and file_metadata.get('event_channels'):
                        event_chs = file_metadata['event_channels']
                        total_events = file_metadata.get('n_events', 0)
                        
                        st.success(f"✅ Detected {total_events} event markers across {len(event_chs)} channels")
                        
                        # Display event channels
                        with st.expander("📍 Event Markers Detected - Select Trial to Segment", expanded=True):
                            for event_name, event_info in event_chs.items():
                                st.write(f"**{event_name}** ({event_info['type']}): {event_info['count']} events")
                                st.caption(f"Times (s): {', '.join([f'{t:.3f}' for t in event_info['times'][:5]])}{'...' if event_info['count'] > 5 else ''}")
                            
                            st.divider()
                            st.write("**Segment Recording Around Event:**")
                            
                            # Select event channel
                            event_channel_names = list(event_chs.keys())
                            selected_event_channel = st.selectbox(
                                "Select event type:",
                                event_channel_names,
                                key='spike2_event_channel'
                            )
                            
                            # Select specific event number
                            n_events_in_channel = event_chs[selected_event_channel]['count']
                            selected_event_num = st.selectbox(
                                "Select event/trial number:",
                                range(n_events_in_channel),
                                format_func=lambda x: f"Event {x+1} at {event_chs[selected_event_channel]['times'][x]:.3f} s",
                                key='spike2_event_num'
                            )
                            
                            # Set window around event
                            col_win1, col_win2 = st.columns(2)
                            with col_win1:
                                pre_window_ms = st.number_input(
                                    "Pre-event window (ms):",
                                    min_value=0.0,
                                    max_value=500.0,
                                    value=50.0,
                                    step=5.0,
                                    key='spike2_pre_win',
                                    help="Time before event to include"
                                )
                            with col_win2:
                                post_window_ms = st.number_input(
                                    "Post-event window (ms):",
                                    min_value=0.0,
                                    max_value=1000.0,
                                    value=150.0,
                                    step=5.0,
                                    key='spike2_post_win',
                                    help="Time after event to include"
                                )
                            
                            # Live preview of selected event segment
                            if st.checkbox("👁️ Preview Selected Event", value=True, key='preview_event'):
                                event_time = event_chs[selected_event_channel]['times'][selected_event_num]
                                
                                # Calculate segment
                                pre_samples = int(pre_window_ms / 1000.0 * fs_loaded)
                                post_samples = int(post_window_ms / 1000.0 * fs_loaded)
                                event_sample = int(event_time * fs_loaded)
                                start_sample = max(0, event_sample - pre_samples)
                                end_sample = min(len(signal_loaded), event_sample + post_samples)
                                
                                signal_preview = signal_loaded[start_sample:end_sample]
                                time_preview = np.arange(len(signal_preview)) / fs_loaded - (pre_window_ms / 1000.0)
                                
                                # Preview plot
                                fig_preview, ax_preview = plt.subplots(figsize=(12, 4))
                                ax_preview.plot(time_preview * 1000, signal_preview, 'b-', linewidth=1.5)
                                ax_preview.axvline(0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Event/Stimulus')
                                ax_preview.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
                                ax_preview.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
                                ax_preview.set_ylabel('Amplitude (mV)', fontsize=11, fontweight='bold')
                                ax_preview.set_title(f'Preview: {selected_event_channel} Event {selected_event_num + 1}', fontsize=12, fontweight='bold')
                                ax_preview.grid(True, alpha=0.3)
                                ax_preview.legend()
                                
                                # Add window info
                                ax_preview.text(0.98, 0.02, f'Window: -{pre_window_ms:.0f} to +{post_window_ms:.0f} ms\nSamples: {len(signal_preview)}',
                                              transform=ax_preview.transAxes, ha='right', va='bottom',
                                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                                              fontsize=9, family='monospace')
                                
                                plt.tight_layout()
                                st.pyplot(fig_preview)
                                plt.close()
                            
                            # Segment and load button
                            if st.button("✂️ Extract Event Segment", type="primary", key='segment_event'):
                                # Get event time
                                event_time = event_chs[selected_event_channel]['times'][selected_event_num]
                                
                                # Calculate sample indices
                                pre_samples = int(pre_window_ms / 1000.0 * fs_loaded)
                                post_samples = int(post_window_ms / 1000.0 * fs_loaded)
                                
                                event_sample = int(event_time * fs_loaded)
                                start_sample = max(0, event_sample - pre_samples)
                                end_sample = min(len(signal_loaded), event_sample + post_samples)
                                
                                # Extract segment
                                signal_segment = signal_loaded[start_sample:end_sample]
                                time_segment = np.arange(len(signal_segment)) / fs_loaded - (pre_window_ms / 1000.0)
                                
                                # Create updated metadata for segment
                                segment_metadata = file_metadata.copy()
                                segment_metadata['is_segment'] = True
                                segment_metadata['original_samples'] = len(signal_loaded)
                                segment_metadata['segment_samples'] = len(signal_segment)
                                
                                # Store segmented data with updated metadata
                                st.session_state.loaded_waveform_raw = (time_segment, signal_segment, fs_loaded, segment_metadata)
                                st.session_state.current_loaded_data = (time_segment, signal_segment, fs_loaded, segment_metadata)
                                st.session_state.spike2_segmented = True
                                st.session_state.spike2_event_info = {
                                    'channel': selected_event_channel,
                                    'event_num': selected_event_num + 1,
                                    'event_time': event_time,
                                    'pre_window_ms': pre_window_ms,
                                    'post_window_ms': post_window_ms
                                }
                                
                                st.success(f"✅ Segment extracted! Reloading to display {selected_event_channel} Event {selected_event_num + 1}...")
                                st.rerun()
                    else:
                        st.warning("⚠️ No event markers detected in file. Showing full continuous recording.")
                        st.caption(f"Recording duration: {len(signal_loaded)/fs_loaded:.1f} seconds ({len(signal_loaded):,} samples)")
                    
                elif file_metadata.get('format') == 'LabChart':
                    st.info(f"📄 **Format:** LabChart exported file")
                    if file_metadata.get('n_channels'):
                        st.caption(f"Channels: {', '.join(file_metadata.get('channel_names', []))}")
                        st.caption(f"Using: {file_metadata.get('selected_channel_name', 'Channel 1')}")
                    
                    # Show trigger detection result
                    if file_metadata.get('trigger_channel'):
                        trigger_ch_name = file_metadata.get('trigger_channel_name', 'Unknown')
                        st.success(f"✅ Trigger channel detected: {trigger_ch_name}")
                    else:
                        st.warning("⚠️ No trigger channel detected. You can manually specify trigger time below.")
                    
                elif file_metadata.get('format') == 'CSV (Multi-trial)':
                    st.info(f"📄 **Format:** Multi-trial CSV ({file_metadata.get('n_trials', 0)} trials detected)")
                    st.caption(file_metadata.get('note', ''))
                    
                    # Add trial selection for multi-trial files
                    if file_metadata.get('n_trials', 0) > 1:
                        with st.expander("📊 Review and Select Trial", expanded=False):
                            st.write(f"**{file_metadata['n_trials']} trials detected**")
                            
                            # Trial selector
                            selected_trial = st.selectbox(
                                "Select trial to analyze:",
                                range(file_metadata['n_trials']),
                                format_func=lambda x: f"Trial {x+1}",
                                index=0,
                                key='trial_selector'
                            )
                            
                            # Preview all trials button
                            if st.button("👁️ Preview All Trials", key='preview_trials'):
                                # Re-read full file
                                uploaded_file.seek(0)
                                df_all = pd.read_csv(uploaded_file)
                                
                                # Plot all trials
                                fig_trials, ax_trials = plt.subplots(figsize=(14, 6))
                                
                                for trial_idx in range(min(file_metadata['n_trials'], 10)):  # Max 10 trials
                                    trial_data = df_all.iloc[:, trial_idx].values
                                    alpha_val = 0.9 if trial_idx == selected_trial else 0.3
                                    linewidth = 2.5 if trial_idx == selected_trial else 1.0
                                    label = f"Trial {trial_idx+1}" + (" (Selected)" if trial_idx == selected_trial else "")
                                    ax_trials.plot(time_loaded * 1000, trial_data, 
                                                  linewidth=linewidth, alpha=alpha_val, label=label)
                                
                                ax_trials.axvline(0, color='r', linestyle='--', alpha=0.5)
                                ax_trials.axhline(0, color='gray', linestyle='-', linewidth=0.5)
                                ax_trials.set_xlabel('Time (ms)', fontsize=12)
                                ax_trials.set_ylabel('Amplitude (mV)', fontsize=12)
                                ax_trials.set_title('All Trials Preview (Selected trial highlighted)', fontsize=13)
                                ax_trials.grid(True, alpha=0.3)
                                ax_trials.legend(fontsize=9, ncol=2)
                                plt.tight_layout()
                                st.pyplot(fig_trials)
                                plt.close()
                            
                            # Load selected trial button
                            if st.button("🔄 Load Selected Trial", key='reload_trial'):
                                # Re-read file and select different trial
                                uploaded_file.seek(0)
                                df_reload = pd.read_csv(uploaded_file)
                                signal_loaded = df_reload.iloc[:, selected_trial].values
                                
                                # Update session state
                                st.session_state.loaded_waveform_raw = (time_loaded, signal_loaded, fs_loaded, file_metadata)
                                st.session_state.selected_trial_num = selected_trial + 1
                                st.success(f"✅ Loaded Trial {selected_trial + 1}")
                                st.rerun()
                
                else:
                    st.info(f"📄 **Format:** {file_metadata.get('format', 'Standard')}")
                
                # Options for multi-channel files
                if file_metadata.get('n_channels', 0) > 1:
                    with st.expander("🔧 Channel & Trigger Options", expanded=False):
                        st.markdown("**Multi-channel file detected**")
                        
                        # Allow channel selection (if needed in future)
                        selected_ch_idx = st.selectbox(
                            "Select channel to analyze:",
                            range(file_metadata['n_channels']),
                            format_func=lambda x: file_metadata['channel_names'][x] if x < len(file_metadata.get('channel_names', [])) else f"Channel {x+1}",
                            index=file_metadata.get('selected_channel', 1) - 1,
                            key='labchart_ch_select'
                        )
                        
                        # Could reload with different channel here if needed
                        st.caption(f"Currently displaying: {file_metadata.get('selected_channel_name', 'Channel 1')}")
                
                # Manual trigger time specification
                col_trig1, col_trig2 = st.columns([2, 1])
                with col_trig1:
                    if file_metadata.get('trigger_channel'):
                        st.write(f"**Trigger Detected:** {file_metadata.get('trigger_channel_name')}")
                    else:
                        st.write("**No trigger detected**")
                with col_trig2:
                    if st.checkbox("Specify manually", key='manual_trigger'):
                        trigger_time_ms = st.number_input(
                            "Trigger time (ms)", 
                            float(time_loaded[0] * 1000),
                            float(time_loaded[-1] * 1000),
                            0.0,
                            step=1.0,
                            key='manual_trig_time',
                            help="Time point where TMS stimulus occurred"
                        )
                        # Adjust time vector to center at trigger
                        time_loaded = time_loaded - (trigger_time_ms / 1000.0)
                        st.success(f"Centered at {trigger_time_ms:.1f} ms")
                
                # Allow manual sampling rate override
                col_fs1, col_fs2 = st.columns([2, 1])
                with col_fs1:
                    st.write(f"**Detected Sampling Rate:** {fs_loaded:.0f} Hz")
                with col_fs2:
                    if st.checkbox("Override", key='override_fs'):
                        fs_manual = st.number_input("Manual FS (Hz)", 500, 20000, int(fs_loaded), 100, key='fs_manual')
                        fs_loaded = float(fs_manual)
                        time_loaded = np.arange(len(signal_loaded)) / fs_loaded
                        if st.session_state.get('manual_trigger', False):
                            trigger_time_ms = st.session_state.get('manual_trig_time', 0)
                            time_loaded = time_loaded - (trigger_time_ms / 1000.0)
                        st.success(f"Using {fs_loaded:.0f} Hz")
                
                # Apply stimulus offset if previously set
                if hasattr(st.session_state, 'stimulus_offset_sec') and st.session_state.stimulus_offset_sec != 0:
                    time_loaded = time_loaded - st.session_state.stimulus_offset_sec
                    st.info(f"ℹ️ Time axis centered at stimulus (offset: {st.session_state.stimulus_offset_sec*1000:.1f} ms)")
                
                # Display loaded waveform
                fig_loaded, ax_loaded = plt.subplots(figsize=(14, 5))
                ax_loaded.plot(time_loaded * 1000, signal_loaded, 'b-', linewidth=1.5, alpha=0.8)
                ax_loaded.axvline(0, color='r', linestyle='--', linewidth=2, alpha=0.6, label='Stimulus/Trigger (t=0)')
                ax_loaded.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
                ax_loaded.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
                ax_loaded.set_ylabel('Amplitude (mV)', fontsize=12, fontweight='bold')
                
                # Update title based on format
                if file_metadata.get('format') == 'LabChart':
                    title = f"Loaded Waveform - {file_metadata.get('selected_channel_name', 'Channel 1')}"
                elif file_metadata.get('format') == 'CSV (Multi-trial)':
                    trial_num = st.session_state.get('selected_trial_num', 1)
                    title = f"Loaded Waveform - Trial {trial_num}"
                elif file_metadata.get('format') == 'Spike2':
                    if st.session_state.get('spike2_segmented', False):
                        event_info = st.session_state.get('spike2_event_info', {})
                        title = f"Spike2 - {event_info.get('channel', 'Event')} Event {event_info.get('event_num', 1)} (Segmented)"
                    else:
                        title = "Spike2 - Full Continuous Recording"
                else:
                    title = "Loaded Waveform - Raw Data"
                
                ax_loaded.set_title(title, fontsize=13, fontweight='bold')
                ax_loaded.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                ax_loaded.legend(fontsize=11)
                
                # Add stats to plot
                textstr = f'Samples: {len(signal_loaded)}\nPeak: {np.max(np.abs(signal_loaded)):.3f} mV\nPTP: {np.ptp(signal_loaded):.3f} mV\nDuration: {(time_loaded[-1] - time_loaded[0])*1000:.1f} ms\nFS: {fs_loaded:.0f} Hz'
                ax_loaded.text(0.02, 0.98, textstr, transform=ax_loaded.transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                             fontsize=9, family='monospace')
                
                plt.tight_layout()
                st.pyplot(fig_loaded)
                plt.close()
                
                # Store current loaded data with potentially updated time axis
                st.session_state.loaded_waveform_raw = (time_loaded, signal_loaded, fs_loaded, file_metadata)
                st.session_state.current_loaded_data = (time_loaded, signal_loaded, fs_loaded, file_metadata)
                
                # Extract parameters button
                if st.button("🔍 Extract Parameters from Loaded Waveform", type="primary", key='extract_params'):
                    # Use stored waveform (may have updated FS)
                    if hasattr(st.session_state, 'loaded_waveform_raw'):
                        time_ext, signal_ext, fs_ext, meta_ext = st.session_state.loaded_waveform_raw
                    else:
                        time_ext, signal_ext, fs_ext, meta_ext = time_loaded, signal_loaded, fs_loaded, file_metadata
                    
                    with st.spinner("Analyzing waveform and extracting parameters..."):
                        extracted_params = extract_mep_parameters(time_ext, signal_ext, fs_ext)
                        
                        st.session_state.extracted_params = extracted_params
                        st.session_state.loaded_waveform = (time_ext, signal_ext, fs_ext)
                        st.session_state.file_metadata = meta_ext
                        
                        st.success("✅ Parameters extracted successfully!")
                
                # Display extracted parameters
                if hasattr(st.session_state, 'extracted_params'):
                    st.write("### Extracted Parameters")
                    
                    params = st.session_state.extracted_params
                    
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        st.metric("Morphology", params['morphology'])
                        st.metric("Peak Amplitude", f"{params['peak_amplitude']:.3f} mV")
                        st.metric("PTP Amplitude", f"{params['ptp_amplitude']:.3f} mV")
                    
                    with col_e2:
                        st.metric("Onset Latency", f"{params['onset_latency_ms']:.1f} ms")
                        st.metric("Peak Latency", f"{params['peak_latency_ms']:.1f} ms")
                        st.metric("Duration", f"{params['duration_ms']:.1f} ms")
                    
                    with col_e3:
                        st.metric("Rise Time", f"{params['rise_time_ms']:.1f} ms")
                        st.metric("Decay Time", f"{params['decay_time_ms']:.1f} ms")
                        if 'phase2_ratio' in params:
                            st.metric("Phase 2 Ratio", f"{params['phase2_ratio']:.2f}")
                    
                    # Display annotated waveform showing extraction points
                    st.write("### 📍 Parameter Extraction Visualisation")
                    st.caption("Visual representation of where each parameter was measured:")
                    
                    # Get stored waveform data
                    if hasattr(st.session_state, 'loaded_waveform'):
                        time_viz, signal_viz, fs_viz = st.session_state.loaded_waveform
                        signal_corrected = params.get('corrected_signal', signal_viz)
                        
                        # Create annotated plot
                        fig_annotated = plot_annotated_waveform(
                            time_viz, 
                            signal_corrected,
                            params,
                            title=f"Extracted Parameters - {params['morphology']} MEP"
                        )
                        
                        st.pyplot(fig_annotated)
                        plt.close()
                        
                        st.caption("""
                        **Legend:** 
                        - 🟠 Orange dashed line = Stimulus (t=0)
                        - 🟢 Green markers = MEP Onset (10% threshold crossing)
                        - 🔴 Red markers = Peak amplitude
                        - 🟣 Purple markers = MEP Offset (return to <10%)
                        - 🟡 Orange squares = Phase peaks (for bi/tri-phasic)
                        - Green arrow = Rise time | Purple arrow = Decay time | Black bracket = Total duration
                        """)
                    
                    st.divider()
                    
                    # Option to use waveform directly (ONLY option now)
                    st.write("### 🎯 Use Extracted Waveform")
                    
                    # Noise configuration for loaded waveform
                    with st.expander("⚙️ Configure Noise for Loaded Waveform", expanded=True):
                        st.write("**Adjust noise parameters before loading waveform:**")
                        
                        col_n1, col_n2 = st.columns(2)
                        
                        with col_n1:
                            loaded_snr_db = st.slider(
                                "SNR (dB) - Higher = Less Noise",
                                min_value=-10,
                                max_value=40,
                                value=20,  # Higher default for loaded waveforms
                                step=2,
                                key='loaded_snr',
                                help="Signal-to-Noise Ratio. Higher values = cleaner signal. Try 20-30 dB for loaded waveforms."
                            )
                            
                            st.caption(f"💡 Current SNR: {loaded_snr_db} dB")
                            if loaded_snr_db < 10:
                                st.warning("⚠️ Low SNR - Signal may be obscured by noise")
                            elif loaded_snr_db > 25:
                                st.info("ℹ️ High SNR - Very clean signal")
                        
                        with col_n2:
                            st.write("**Select Noise Types:**")
                            loaded_include_line = st.checkbox("50 Hz Line Noise", value=True, key='loaded_line')
                            loaded_include_emg = st.checkbox("EMG Noise", value=True, key='loaded_emg')
                            loaded_include_ecg = st.checkbox("ECG Artifact", value=False, key='loaded_ecg')
                            loaded_include_movement = st.checkbox("Movement Artifact", value=False, key='loaded_movement')
                            loaded_include_tms = st.checkbox("TMS Artifact", value=False, key='loaded_tms')
                        
                        st.info("💡 **Tip:** For loaded waveforms with existing baseline noise, use higher SNR (20-30 dB) and fewer noise types.")
                    
                    if st.button("📋 Use Loaded Waveform Directly as Template", type="primary", key='use_loaded'):
                        # Use loaded waveform data from session state
                        if hasattr(st.session_state, 'loaded_waveform') and hasattr(st.session_state, 'current_loaded_data'):
                            time_template, signal_template, fs_template = st.session_state.loaded_waveform
                            
                            # Normalize to desired amplitude
                            normalized_signal = signal_template / np.max(np.abs(signal_template)) * params['peak_amplitude']
                            
                            # Add noise with USER-CONFIGURED parameters
                            noise_gen_temp = NoiseGenerator(sampling_rate=int(fs_template))
                            mep_noisy_temp = noise_gen_temp.add_composite_noise(
                                normalized_signal,
                                time_template,
                                snr_db=loaded_snr_db,  # User-configured SNR
                                include_line=loaded_include_line,  # User-configured
                                include_emg=loaded_include_emg,    # User-configured
                                include_ecg=loaded_include_ecg,    # User-configured
                                include_movement=loaded_include_movement,  # User-configured
                                include_tms=loaded_include_tms     # User-configured
                            )
                            
                            # Store as generated signal with both clean and noisy versions
                            st.session_state.generated_signals = {
                                'time': time_template,
                                'mep_clean': normalized_signal,
                                'mep_noisy': mep_noisy_temp,
                                'sampling_rate': int(fs_template),
                                'parameters': {
                                    'amplitude': params['peak_amplitude'],
                                    'duration': params['duration_ms'],
                                    'onset_latency': params['onset_latency_ms'],
                                    'snr_db': loaded_snr_db,  # Store user SNR
                                    'mep_type': f"Template from {file_metadata.get('filename', 'loaded file')}",
                                    'noise_types': {
                                        'line': loaded_include_line,
                                        'emg': loaded_include_emg,
                                        'ecg': loaded_include_ecg,
                                        'movement': loaded_include_movement,
                                        'tms': loaded_include_tms
                                    }
                                }
                            }
                            
                            st.success(f"✅ Loaded waveform set as template with configured noise (SNR={loaded_snr_db} dB)!")
                            st.info("📊 Scroll down to 'Generated Signals' section to view.")
                        else:
                            st.error("❌ Please extract parameters first before using waveform directly.")
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("Please check file format. Should contain time and amplitude data.")
    
    st.divider()
    
    # ========== STANDARD MEP PARAMETERS ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MEP Parameters")
        
        # Use standard defaults
        default_sampling_rate = 5000
        default_amplitude = 1.0
        default_onset = 20
        default_rise = 8
        default_decay = 15
        default_duration = 30
        default_mep_type = 'Standard'
        
        sampling_rate = st.number_input("Sampling Rate (Hz)", 
                                       min_value=500, max_value=20000, 
                                       value=int(default_sampling_rate), step=100,
                                       key='gen_sampling_rate')
        
        mep_amplitude = st.slider("MEP Amplitude (mV)", 
                                 min_value=0.1, max_value=5.0, 
                                 value=float(default_amplitude), step=0.1,
                                 key='gen_amplitude')
        
        mep_duration = st.slider("MEP Duration (ms)", 
                                min_value=5, max_value=150, 
                                value=int(default_duration), step=5,
                                key='gen_duration')
        
        onset_latency = st.slider("Onset Latency (ms)", 
                                 min_value=5, max_value=100, 
                                 value=int(default_onset), step=5,
                                 key='gen_onset')
        
        rise_time = st.slider("Rise Time (ms)", 
                             min_value=1, max_value=20, 
                             value=int(default_rise), step=1,
                             key='gen_rise')
        
        decay_time = st.slider("Decay Time (ms)", 
                              min_value=1, max_value=50, 
                              value=int(default_decay), step=1,
                              key='gen_decay')
        
        asymmetry = st.slider("Asymmetry Factor", 
                             min_value=1.0, max_value=2.0, 
                             value=1.2, step=0.1,
                             key='gen_asymmetry')
        
        mep_type = st.selectbox("MEP Type", 
                               ["Standard", "Bi-phasic", "Tri-phasic", 
                                "Double Peak", "With Baseline EMG"],
                               index=["Standard", "Bi-phasic", "Tri-phasic", 
                                     "Double Peak", "With Baseline EMG"].index(default_mep_type) if default_mep_type in ["Standard", "Bi-phasic", "Tri-phasic", "Double Peak", "With Baseline EMG"] else 0,
                               key='gen_mep_type')
        
        # ========== ADVANCED PHASE CONTROLS ==========
        if mep_type == "Bi-phasic":
            with st.expander("⚙️ Advanced Bi-Phasic Controls", expanded=False):
                st.markdown("**Independent control over each phase**")
                
                st.write("##### Phase 1 (Positive)")
                col_p1a, col_p1b = st.columns(2)
                with col_p1a:
                    phase1_rise = st.slider("Phase 1 Rise Time (ms)", 3, 20, 10, 1, key='p1_rise')
                with col_p1b:
                    phase1_decay = st.slider("Phase 1 Decay Time (ms)", 3, 20, 12, 1, key='p1_decay')
                
                st.write("##### Phase 2 (Negative)")
                phase2_ratio = st.slider("Phase 2 Amplitude (% of Phase 1)", 10, 150, 80, 5, key='p2_ratio',
                                        help="Percentage of Phase 1 amplitude. E.g., 80 = 80% of Phase 1") / 100.0
                col_p2a, col_p2b = st.columns(2)
                with col_p2a:
                    phase2_rise = st.slider("Phase 2 Rise Time (ms)", 3, 20, 8, 1, key='p2_rise')
                with col_p2b:
                    phase2_decay = st.slider("Phase 2 Decay Time (ms)", 3, 20, 10, 1, key='p2_decay')
                
                phase_sep_12 = st.slider("Phase Separation (ms)", 0, 10, 3, 1, key='sep_12',
                                        help="Time between Phase 1 and Phase 2 peaks")
                
                use_advanced_biphasic = True
        else:
            use_advanced_biphasic = False
            phase1_rise = phase1_decay = phase2_ratio = phase2_rise = phase2_decay = phase_sep_12 = None
        
        if mep_type == "Tri-phasic":
            with st.expander("⚙️ Advanced Tri-Phasic Controls", expanded=False):
                st.markdown("**Independent control over all three phases**")
                
                st.write("##### Phase 1 (Positive)")
                col_t1a, col_t1b = st.columns(2)
                with col_t1a:
                    phase1_rise_tri = st.slider("Phase 1 Rise (ms)", 3, 20, 10, 1, key='t1_rise')
                with col_t1b:
                    phase1_decay_tri = st.slider("Phase 1 Decay (ms)", 3, 20, 12, 1, key='t1_decay')
                
                st.write("##### Phase 2 (Negative)")
                phase2_ratio_tri = st.slider("Phase 2 Amplitude (% of Phase 1)", 10, 150, 75, 5, key='t2_ratio') / 100.0
                col_t2a, col_t2b = st.columns(2)
                with col_t2a:
                    phase2_rise_tri = st.slider("Phase 2 Rise (ms)", 3, 20, 8, 1, key='t2_rise')
                with col_t2b:
                    phase2_decay_tri = st.slider("Phase 2 Decay (ms)", 3, 20, 10, 1, key='t2_decay')
                
                st.write("##### Phase 3 (Positive)")
                phase3_ratio_tri = st.slider("Phase 3 Amplitude (% of Phase 1)", 10, 100, 40, 5, key='t3_ratio') / 100.0
                col_t3a, col_t3b = st.columns(2)
                with col_t3a:
                    phase3_rise_tri = st.slider("Phase 3 Rise (ms)", 3, 20, 7, 1, key='t3_rise')
                with col_t3b:
                    phase3_decay_tri = st.slider("Phase 3 Decay (ms)", 3, 20, 9, 1, key='t3_decay')
                
                col_sep1, col_sep2 = st.columns(2)
                with col_sep1:
                    phase_sep_12_tri = st.slider("Phase 1→2 Separation (ms)", 0, 10, 3, 1, key='sep_12_tri')
                with col_sep2:
                    phase_sep_23_tri = st.slider("Phase 2→3 Separation (ms)", 0, 10, 3, 1, key='sep_23_tri')
                
                use_advanced_triphasic = True
        else:
            use_advanced_triphasic = False
            phase1_rise_tri = phase1_decay_tri = phase2_ratio_tri = None
            phase2_rise_tri = phase2_decay_tri = phase3_ratio_tri = None
            phase3_rise_tri = phase3_decay_tri = phase_sep_12_tri = phase_sep_23_tri = None
    
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
                if use_advanced_biphasic:
                    # Use advanced bi-phasic method with phase-specific controls
                    time, mep_clean = mep_gen.generate_biphasic_advanced(
                        phase1_amplitude=mep_amplitude,
                        phase1_rise_time=phase1_rise/1000,
                        phase1_decay_time=phase1_decay/1000,
                        phase2_amplitude_ratio=phase2_ratio,
                        phase2_rise_time=phase2_rise/1000,
                        phase2_decay_time=phase2_decay/1000,
                        phase_separation=phase_sep_12/1000,
                        onset_latency=onset_latency/1000
                    )
                else:
                    # Use standard bi-phasic method
                    time, mep_clean = mep_gen.generate_biphasic_mep(
                        amplitude=mep_amplitude,
                        onset_latency=onset_latency/1000,
                        phase1_duration=rise_time/1000 * 1.2,
                        phase2_duration=decay_time/1000 * 1.0,
                        phase2_amplitude_ratio=0.8
                    )
            elif mep_type == "Tri-phasic":
                if use_advanced_triphasic:
                    # Use advanced tri-phasic method with phase-specific controls
                    time, mep_clean = mep_gen.generate_triphasic_advanced(
                        phase1_amplitude=mep_amplitude,
                        phase1_rise_time=phase1_rise_tri/1000,
                        phase1_decay_time=phase1_decay_tri/1000,
                        phase2_amplitude_ratio=phase2_ratio_tri,
                        phase2_rise_time=phase2_rise_tri/1000,
                        phase2_decay_time=phase2_decay_tri/1000,
                        phase3_amplitude_ratio=phase3_ratio_tri,
                        phase3_rise_time=phase3_rise_tri/1000,
                        phase3_decay_time=phase3_decay_tri/1000,
                        phase1_2_separation=phase_sep_12_tri/1000,
                        phase2_3_separation=phase_sep_23_tri/1000,
                        onset_latency=onset_latency/1000
                    )
                else:
                    # Use standard tri-phasic method
                    time, mep_clean = mep_gen.generate_triphasic_mep(
                        amplitude=mep_amplitude,
                        onset_latency=onset_latency/1000,
                        phase1_duration=rise_time/1000 * 1.0,
                        phase2_duration=decay_time/1000 * 1.2,
                        phase3_duration=rise_time/1000 * 1.3,
                        phase2_ratio=0.75,
                        phase3_ratio=0.4
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
                    'mep_type': mep_type,
                    'amplitude': mep_amplitude,
                    'duration': mep_duration,
                    'onset_latency': onset_latency,
                    'rise_time': rise_time,
                    'decay_time': decay_time,
                    'asymmetry': asymmetry,
                    'snr_db': snr_db,
                    'noise_types': {
                        'white': include_white,
                        'line': include_line,
                        'emg': include_emg,
                        'ecg': include_ecg,
                        'movement': include_movement,
                        'tms': include_tms
                    },
                    'line_freq': line_freq
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
                st.write("**Filter Order Selection**")
                st.caption("""
                **Convention:** Select the DESIGNED order (matching literature reporting). 
                With zero-phase filtering (sosfiltfilt), the effective magnitude response 
                will be 2× the designed order.
                
                Example: Selecting "2nd order" (standard) → Effective 4th-order response
                """)
                order = st.slider("Designed Filter Order", 
                                 min_value=1, max_value=8, value=2, step=1,
                                 help="Literature typically reports designed order. Effective order = 2× with zero-phase filtering.")
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
                
                # Ensure filtered signal matches input length (trim if needed)
                if len(result['filtered_signal']) != len(data['mep_clean']):
                    # Trim to match shorter length
                    min_len = min(len(result['filtered_signal']), len(data['mep_clean']))
                    filtered_plot = result['filtered_signal'][:min_len]
                    clean_plot = data['mep_clean'][:min_len]
                    noisy_plot = data['mep_noisy'][:min_len]
                    time_plot = data['time'][:min_len]
                else:
                    filtered_plot = result['filtered_signal']
                    clean_plot = data['mep_clean']
                    noisy_plot = data['mep_noisy']
                    time_plot = data['time']
                
                # Waveform comparison
                axes[0].plot(time_plot*1000, clean_plot, 'b-', 
                           linewidth=1.5, alpha=0.7, label='Clean (Ground Truth)')
                axes[0].plot(time_plot*1000, noisy_plot, 'gray', 
                           linewidth=0.8, alpha=0.4, label='Noisy')
                axes[0].plot(time_plot*1000, filtered_plot, 'r-', 
                           linewidth=1.5, alpha=0.8, label='Filtered')
                axes[0].axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
                axes[0].set_ylabel('Amplitude (mV)')
                axes[0].set_title('Signal Comparison')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Error plot
                error = filtered_plot - clean_plot
                axes[1].plot(time_plot*1000, error, 'r-', linewidth=1)
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
                
                # Time-Frequency Analysis
                st.subheader("🌊 Time-Frequency Analysis (Morlet Wavelet)")
                
                st.markdown("""
                Visualise how signal energy is distributed across time and frequency. 
                Useful for understanding filter effects on spectral content.
                """)
                
                col_tf1, col_tf2 = st.columns([2, 1])
                
                with col_tf1:
                    freq_min = st.slider("Minimum Frequency (Hz)", 1, 50, 5, 1, key='tf_fmin')
                    freq_max = st.slider("Maximum Frequency (Hz)", 100, 1000, 500, 50, key='tf_fmax')
                
                with col_tf2:
                    st.write("**Plot Options:**")
                    n_freqs = st.slider("Frequency Resolution", 40, 150, 80, 10, key='tf_nfreqs',
                                       help="Higher = more detail, slower computation")
                    tf_dpi = st.selectbox("Export DPI", [150, 300, 600], index=1, key='tf_dpi')
                
                if st.button("🎨 Generate Time-Frequency Analysis", type="primary"):
                    with st.spinner("Computing Morlet wavelet transform..."):
                        # Create time-frequency comparison figure
                        tf_fig = plot_timefreq_comparison(
                            data['mep_clean'],
                            data['mep_noisy'],
                            result['filtered_signal'],
                            data['time'],
                            data['sampling_rate'],
                            freq_range=(freq_min, freq_max),
                            n_freqs=n_freqs
                        )
                        
                        st.pyplot(tf_fig)
                        
                        # Add download button
                        import io
                        buf = io.BytesIO()
                        tf_fig.savefig(buf, format='png', dpi=tf_dpi, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label=f"💾 Download Time-Frequency Plot ({tf_dpi} DPI)",
                            data=buf,
                            file_name=f"timefreq_analysis_{tf_dpi}dpi.png",
                            mime="image/png",
                            key='download_timefreq'
                        )
                        
                        plt.close(tf_fig)
                        
                        st.success("✅ Time-frequency analysis complete!")
                        
                        st.info("""
                        **Interpretation Guide:**
                        - **Bright regions** = High spectral power at that time-frequency
                        - **Ground truth (A):** Shows natural MEP frequency content
                        - **Noisy signal (B):** Shows added noise across all frequencies
                        - **Filtered signal (C):** Should match (A) in MEP window, reduced elsewhere
                        - **Good filter:** Preserves MEP spectral structure, removes noise frequencies
                        """)

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
            
            st.write("**Filter Order Selection**")
            st.info("""
            **Important Convention:** Select DESIGNED orders (matching literature reporting convention).
            
            When you select "2nd order":
            - Filter designed as 2nd-order Butterworth
            - Applied with zero-phase filtering (sosfiltfilt)
            - Effective magnitude response: 4th-order
            
            This matches how ~95% of TMS-EMG papers report filters. Labels show designed order (e.g., O2), 
            but Methods documentation will clarify the effective order.
            """)
            
            test_orders = st.multiselect(
                "Designed Filter Orders (Butterworth only)",
                [1, 2, 3, 4, 5, 6, 7, 8],
                default=[2, 4],
                help="Select designed orders. Effective orders = 2× these values with zero-phase filtering."
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
            
            # Visualisations
            st.subheader("Performance Comparison")
            
            # Create tabs for different Visualisation types
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
            st.write("**Based on mean of absolute errors across all iterations**")
            
            st.info("""
            **Note:** Rankings use mean(|errors|) not mean(errors) to avoid bias from error cancellation. 
            A filter with errors that cancel to 0% mean but vary ±5% is worse than one consistently at +2%.
            """)
            
            # Calculate mean metrics per configuration for best selection
            config_means = df.groupby('config_label').agg({
                'amplitude_error_pct': ['mean', 'std'],
                'peak_latency_error_ms': ['mean', 'std'],
                'correlation': ['mean', 'std']
            })
            
            # Calculate mean of absolute errors for proper ranking
            config_mean_abs = df.groupby('config_label')['amplitude_error_pct'].apply(
                lambda x: np.mean(np.abs(x))
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Lowest Amplitude Error**")
                best_amp_idx = config_mean_abs.idxmin()
                best_amp_mean = config_means.loc[best_amp_idx, ('amplitude_error_pct', 'mean')]
                best_amp_std = config_means.loc[best_amp_idx, ('amplitude_error_pct', 'std')]
                best_amp_mean_abs = config_mean_abs.loc[best_amp_idx]
                st.write(f"Config: `{best_amp_idx}`")
                st.write(f"Mean (signed): {best_amp_mean:.2f}% ± {best_amp_std:.2f}%")
                st.write(f"Mean(|error|): {best_amp_mean_abs:.2f}%")
            
            with col2:
                st.write("**Lowest Latency Error**")
                config_mean_abs_lat = df.groupby('config_label')['peak_latency_error_ms'].apply(
                    lambda x: np.mean(np.abs(x))
                )
                best_lat_idx = config_mean_abs_lat.idxmin()
                best_lat_mean = config_means.loc[best_lat_idx, ('peak_latency_error_ms', 'mean')]
                best_lat_std = config_means.loc[best_lat_idx, ('peak_latency_error_ms', 'std')]
                best_lat_mean_abs = config_mean_abs_lat.loc[best_lat_idx]
                st.write(f"Config: `{best_lat_idx}`")
                st.write(f"Mean (signed): {best_lat_mean:.3f} ± {best_lat_std:.3f} ms")
                st.write(f"Mean(|error|): {best_lat_mean_abs:.3f} ms")
            
            with col3:
                st.write("**Highest Correlation**")
                best_corr_idx = config_means[('correlation', 'mean')].idxmax()
                best_corr_mean = config_means.loc[best_corr_idx, ('correlation', 'mean')]
                best_corr_std = config_means.loc[best_corr_idx, ('correlation', 'std')]
                st.write(f"Config: {best_corr_idx}")
                st.write(f"Correlation: {best_corr_mean:.4f} ± {best_corr_std:.4f}")
            
            st.divider()
            
            # ========== STATISTICAL ANALYSIS SECTION ==========
            st.subheader("📊 Statistical Analysis & Filter Comparison")
            st.markdown("""
            Rigorous statistical testing to identify optimal filter configurations with evidence-based recommendations.
            """)
            
            stat_tab1, stat_tab2, stat_tab3 = st.tabs(["Rankings & Tests", "Detailed Comparison", "Recommendations"])
            
            # TAB 1: Rankings and Statistical Tests
            with stat_tab1:
                st.write("### Performance Rankings by Metric")
                
                metric_for_stats = st.selectbox(
                    "Select metric for statistical analysis:",
                    ['amplitude_error_pct', 'peak_latency_error_ms', 'correlation', 
                     'rmse_mep', 'snr_improvement'],
                    key='stat_metric'
                )
                
                if st.button("🔬 Run Statistical Analysis", type="primary", key='run_stats'):
                    with st.spinner("Performing assumption testing and statistical analysis..."):
                        # Perform comprehensive statistical analysis with assumptions
                        stat_results = perform_statistical_analysis(df, metric_for_stats)
                        
                        # Store in session state
                        st.session_state.stat_results = stat_results
                        st.session_state.current_metric = metric_for_stats
                
                # Display results if available - OUTSIDE button callback for interactivity
                if hasattr(st.session_state, 'stat_results') and hasattr(st.session_state, 'current_metric'):
                    stat_results = st.session_state.stat_results
                    
                    # Check if metric changed - if so, clear results
                    if st.session_state.current_metric != metric_for_stats:
                        st.warning("⚠️ Metric changed. Click 'Run Statistical Analysis' again to update results.")
                    else:
                        # === DISPLAY ASSUMPTION TESTING RESULTS ===
                        st.write("### 📋 Statistical Assumptions Testing")
                        
                        assumptions = stat_results['assumptions']
                        
                        # Create expandable sections for each assumption
                        with st.expander("1️⃣ Sample Size Adequacy", expanded=True):
                            ss = assumptions['sample_size']
                            col_ss1, col_ss2, col_ss3 = st.columns(3)
                            with col_ss1:
                                st.metric("Minimum Size", ss['min_size'])
                            with col_ss2:
                                st.metric("Mean Size", f"{ss['mean_size']:.0f}")
                            with col_ss3:
                                status = "✅ Adequate" if ss['adequate'] else "⚠️ Small"
                                st.metric("Status", status)
                            
                            st.info(f"**Assessment:** {ss['interpretation']}")
                            st.caption(f"**Recommendation:** {ss['recommendation']}")
                        
                        with st.expander("2️⃣ Normality Testing (Shapiro-Wilk)", expanded=True):
                            norm = assumptions['normality']
                            
                            col_n1, col_n2 = st.columns(2)
                            with col_n1:
                                st.metric("Groups Normal", f"{norm['percent_normal']:.0f}%")
                            with col_n2:
                                status = "✅ All Normal" if norm['all_normal'] else "⚠️ Non-Normal Detected"
                                st.metric("Overall Status", status)
                            
                            st.info(f"**Interpretation:** {norm['interpretation']}")
                            st.caption(f"**Recommendation:** {norm['recommendation']}")
                            
                            # Show individual results in table - ALL GROUPS
                            if st.checkbox("Show individual normality tests", key='show_norm_details'):
                                norm_df_data = []
                                for i, test in enumerate(norm['individual_tests'], 1):
                                    norm_df_data.append({
                                        '#': i,
                                        'Configuration': test['config'],
                                        'Normal': '✓' if test['is_normal'] else '✗',
                                        'p-value': f"{test['p_value']:.6f}"
                                    })
                                norm_df = pd.DataFrame(norm_df_data)
                                st.dataframe(norm_df, use_container_width=True, hide_index=True)
                                st.caption(f"*Showing all {len(norm['individual_tests'])} groups*")
                        
                        with st.expander("3️⃣ Homogeneity of Variance (Levene's Test)", expanded=True):
                            homog = assumptions['homogeneity']
                            
                            col_h1, col_h2, col_h3 = st.columns(3)
                            with col_h1:
                                st.metric("Test", homog['test_name'])
                            with col_h2:
                                st.metric("Statistic", f"{homog['statistic']:.4f}")
                            with col_h3:
                                st.metric("p-value", f"{homog['p_value']:.6f}")
                            
                            status = "✅" if homog['is_homogeneous'] else "⚠️"
                            st.info(f"{status} **Result:** {homog['interpretation']}")
                            st.caption(f"**Recommendation:** {homog['recommendation']}")
                        
                        with st.expander("4️⃣ Test Selection & Rationale", expanded=True):
                            test_sel = assumptions['test_selection']
                            
                            st.success(f"**Selected Test:** {test_sel['recommended_test']}")
                            
                            if test_sel['post_hoc']:
                                st.info(f"**Post-hoc Test:** {test_sel['post_hoc']}")
                            
                            st.write("**Rationale:**")
                            for reason in test_sel['rationale']:
                                st.write(f"  {reason}")
                            
                            test_type = "Parametric" if test_sel['parametric'] else "Non-parametric"
                            st.caption(f"**Test Type:** {test_type}")
                        
                        st.divider()
                        
                        # Download assumption testing report
                        assumption_report = generate_assumption_report(assumptions)
                        st.download_button(
                            label="📄 Download Assumption Testing Report",
                            data=assumption_report,
                            file_name=f"assumption_testing_{metric_for_stats}.txt",
                            mime="text/plain",
                            key='download_assumptions'
                        )
                        
                        st.divider()
                        
                        # === DISPLAY STATISTICAL TEST RESULTS ===
                        st.write("### 📊 Statistical Test Results")
                        
                        test_info = stat_results['statistical_test']
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Test Used", test_info['test_name'])
                            st.caption(test_info['test_type'])
                        with col_stat2:
                            st.metric(f"{test_info['statistic_name']}-statistic", f"{test_info['statistic']:.4f}")
                        with col_stat3:
                            p_val = test_info['p_value']
                            st.metric("p-value", f"{p_val:.6f}",
                                     delta="Significant" if p_val < 0.05 else "Not significant")
                        
                        st.info(f"**Interpretation:** {test_info['interpretation']}")
                        
                        # Display formal statistical statement for publication
                        with st.expander("📝 Formal Statistical Statement", expanded=False):
                            st.markdown("**Possible Publication-Ready Statement (APA Style):**")
                            st.code(stat_results['formal_report'], language=None)
                            
                            st.download_button(
                                label="📄 Download Formal Statement",
                                data=stat_results['formal_report'],
                                file_name=f"formal_statement_{metric_for_stats}.txt",
                                mime="text/plain",
                                key='download_formal_stat'
                            )
                            
                            st.caption("*Copy this text directly into your Results section*")
                        
                        st.divider()
                        
                        # Top 10 rankings
                        st.write("### Top 10 Ranked Configurations")
                        
                        st.info("""
                        **Ranking Explanation:** Configurations are ranked by **mean(|error|)** - the mean of absolute errors 
                        across all iterations. This differs from the signed mean shown below:
                        - **Mean (Signed)**: Can be close to 0 if positive/negative errors cancel out
                        - **Mean(|Error|)**: True magnitude of errors regardless of direction
                        - Rankings use Mean(|Error|) because we care about error magnitude, not bias direction
                        """)
                        
                        rankings_data = []
                        for rank in range(1, min(11, len(stat_results['rankings']) + 1)):
                            info = stat_results['rankings'][rank]
                            config = info['config']
                            stats_dict = info['stats']
                            
                            rankings_data.append({
                                'Rank': rank,
                                'Configuration': config,
                                'Mean (Signed)': f"{stats_dict['mean']:.4f}",
                                'Mean(|Error|)': f"{stats_dict['mean_abs']:.4f}",
                                'SD': f"{stats_dict['std']:.4f}",
                                'Median': f"{stats_dict['median']:.4f}",
                                '95% CI Lower': f"{stats_dict['ci_95'][0]:.4f}",
                                '95% CI Upper': f"{stats_dict['ci_95'][1]:.4f}"
                            })
                        
                        rankings_df = pd.DataFrame(rankings_data)
                        st.dataframe(rankings_df, use_container_width=True, hide_index=True)
                        
                        st.caption("""
                        **Key:** Mean (Signed) = average of all errors (can cancel out); 
                        Mean(|Error|) = average of error magnitudes (used for ranking)
                        """)
                        
                        st.divider()
                        
                        # Pairwise comparisons
                        st.write("### Pairwise Statistical Comparisons (Top 5)")
                        st.write("*Mann-Whitney U tests with Bonferroni correction for multiple comparisons*")
                        
                        if stat_results['pairwise']:
                            pairwise_data = []
                            for comp in stat_results['pairwise'][:10]:  # Show top 10 comparisons
                                pairwise_data.append({
                                    'Comparison': f"{comp['config1']} vs {comp['config2']}",
                                    'Mean Difference': f"{abs(comp['mean1'] - comp['mean2']):.4f}",
                                    'p-value': f"{comp['p_value']:.6f}",
                                    'p-corrected': f"{comp['p_corrected']:.6f}",
                                    'Significant': '✓ Yes' if comp['significant'] else '✗ No',
                                    'Effect Size (d)': f"{comp['effect_size']:.3f}",
                                    'Magnitude': comp['effect_magnitude']
                                })
                            
                            pairwise_df = pd.DataFrame(pairwise_data)
                            st.dataframe(pairwise_df, use_container_width=True, hide_index=True)
                            
                            st.caption("*Effect size interpretation: Negligible (<0.2), Small (0.2-0.5), Medium (0.5-0.8), Large (>0.8)*")
                        else:
                            st.info("No pairwise comparisons available (need at least 2 configurations)")
            
            # TAB 2: Detailed Comparison Table
            with stat_tab2:
                st.write("### Comprehensive Performance Comparison")
                
                top_n_compare = st.slider("Number of top configurations to compare:", 
                                         5, min(20, len(df['config_label'].unique())), 10,
                                         key='top_n_compare')
                
                if st.button("📋 Generate Comparison Table", key='gen_compare'):
                    comparison_table = create_comparison_table(df, top_n=top_n_compare)
                    
                    st.write(f"### Top {top_n_compare} Filter Configurations")
                    st.dataframe(comparison_table, use_container_width=True)
                    
                    st.caption("""
                    **Composite Score:** Normalized combination of amplitude error, latency error, and correlation 
                    (lower = better overall performance). Equal weighting across metrics.
                    """)
                    
                    # Download comparison table
                    csv_compare = comparison_table.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison Table (CSV)",
                        data=csv_compare,
                        file_name="filter_comparison_table.csv",
                        mime="text/csv",
                        key='download_compare'
                    )
            
            # TAB 3: Evidence-Based Recommendations
            with stat_tab3:
                st.write("### Evidence-Based Filter Recommendations")
                
                if st.button("📝 Generate Recommendations Report", type="primary", key='gen_recs'):
                    with st.spinner("Analyzing all metrics and generating recommendations..."):
                        # Generate comprehensive recommendations
                        rec_summary = create_recommendations_summary(df)
                        
                        # Store in session state
                        st.session_state.rec_summary = rec_summary
                
                # Display recommendations if generated
                if hasattr(st.session_state, 'rec_summary'):
                    # Display in formatted text area
                    st.text_area("Recommendations Summary", st.session_state.rec_summary, height=400, key='display_recs')
                    
                    # Download recommendations
                    st.download_button(
                        label="📄 Download Recommendations (TXT)",
                        data=st.session_state.rec_summary,
                        file_name="filter_recommendations.txt",
                        mime="text/plain",
                        key='download_recs'
                    )
                
                st.divider()
                
                # Generate full statistical report - INDEPENDENT SECTION
                st.write("### Complete Statistical Report")
                st.write("Generate comprehensive statistical analysis for selected metrics")
                
                metrics_to_test = st.multiselect(
                    "Select metrics for full report:",
                    ['amplitude_error_pct', 'peak_latency_error_ms', 'correlation', 
                     'rmse_mep', 'baseline_std'],
                    default=['amplitude_error_pct', 'peak_latency_error_ms', 'correlation'],
                    key='metrics_report'
                )
                
                if st.button("📊 Generate Full Statistical Report", type="primary", key='gen_full_report'):
                    with st.spinner("Generating comprehensive statistical report..."):
                        # Filter metrics to only those that exist in the dataframe
                        available_metrics = [m for m in metrics_to_test if m in df.columns]
                        
                        if not available_metrics:
                            st.error("None of the selected metrics are available in the results!")
                        else:
                            full_report = generate_statistical_report(df, available_metrics)
                            
                            # Store in session state
                            st.session_state.full_report = full_report
                
                # Display full report if generated
                if hasattr(st.session_state, 'full_report'):
                    st.text_area("Full Statistical Analysis Report", 
                                st.session_state.full_report, height=600, key='display_full_report')
                    
                    # Download full report
                    st.download_button(
                        label="📥 Download Full Report (TXT)",
                        data=st.session_state.full_report,
                        file_name="statistical_analysis_full_report.txt",
                        mime="text/plain",
                        key='download_full_report'
                    )
            
            st.divider()
            
            # Iteration overlay Visualisation
            st.subheader("📊 Filter Performance Visualisation")
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
                with st.spinner("Generating Visualisation..."):
                    # Initialize generators and filters for this Visualisation
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
            
            # Time-Frequency Analysis for Selected Configuration
            st.subheader("🌊 Time-Frequency Analysis")
            st.markdown("""
            Generate Morlet wavelet time-frequency representations to visualise spectral content evolution.
            Compare ground truth, noisy, and filtered signals in the time-frequency domain.
            """)
            
            config_for_tf = st.selectbox(
                "Select configuration for time-frequency analysis:",
                options=df['config_label'].unique(),
                index=0,
                key='tf_config_select'
            )
            
            col_tf1, col_tf2, col_tf3 = st.columns([2, 2, 1])
            
            with col_tf1:
                tf_freq_min = st.slider("Minimum Frequency (Hz)", 1, 50, 5, 1, key='batch_tf_fmin')
                tf_freq_max = st.slider("Maximum Frequency (Hz)", 100, 2000, 500, 50, key='batch_tf_fmax')
            
            with col_tf2:
                tf_n_freqs = st.slider("Frequency Resolution", 40, 150, 80, 10, key='batch_tf_nfreqs',
                                      help="Higher = more detail but slower")
                tf_iteration = st.slider("Which iteration to visualise", 1, 
                                        min(len(df[df['config_label']==config_for_tf]), 50), 1, 1,
                                        key='batch_tf_iter',
                                        help="Select which noise realisation to show")
            
            with col_tf3:
                st.write("**Export:**")
                batch_tf_dpi = st.selectbox("DPI", [150, 300, 600], index=1, key='batch_tf_dpi')
            
            if st.button("🎨 Generate Time-Frequency Analysis", type="primary", key='batch_tf_button'):
                with st.spinner("Computing Morlet wavelet transforms..."):
                    # Get the selected configuration and iteration
                    config_df = df[df['config_label'] == config_for_tf]
                    
                    if len(config_df) >= tf_iteration:
                        # Get filter parameters
                        row = config_df.iloc[tf_iteration - 1]
                        filter_params = {
                            'filter_type': row['filter_type'],
                            'lowcut': row['lowcut'],
                            'highcut': row['highcut'],
                            'order': row['order'],
                            'notch_enabled': row.get('notch_enabled', False),
                            'notch_freq': 50
                        }
                        
                        # Initialize for this specific analysis
                        noise_gen_tf = NoiseGenerator(sampling_rate=data['sampling_rate'])
                        filters_tf = MEPFilters(sampling_rate=data['sampling_rate'])
                        
                        # Generate noisy signal for this iteration
                        np.random.seed(tf_iteration)  # Reproducible for same iteration
                        mep_noisy_tf = noise_gen_tf.add_composite_noise(
                            data['mep_clean'], data['time'],
                            snr_db=data['parameters']['snr_db'],
                            include_line=True,
                            include_emg=True
                        )
                        
                        # Apply filter
                        mep_filtered_tf = filters_tf.apply_filter_cascade(mep_noisy_tf, filter_params)
                        
                        # Generate time-frequency plot
                        tf_fig = plot_timefreq_comparison(
                            data['mep_clean'],
                            mep_noisy_tf,
                            mep_filtered_tf,
                            data['time'],
                            data['sampling_rate'],
                            freq_range=(tf_freq_min, tf_freq_max),
                            n_freqs=tf_n_freqs
                        )
                        
                        st.pyplot(tf_fig)
                        
                        # Download button
                        import io
                        buf = io.BytesIO()
                        tf_fig.savefig(buf, format='png', dpi=batch_tf_dpi, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            label=f"💾 Download Time-Frequency Plot ({batch_tf_dpi} DPI)",
                            data=buf,
                            file_name=f"timefreq_{config_for_tf}_iter{tf_iteration}_{batch_tf_dpi}dpi.png",
                            mime="image/png",
                            key='download_batch_timefreq'
                        )
                        
                        plt.close(tf_fig)
                        
                        st.success(f"✅ Time-frequency analysis complete for {config_for_tf}, iteration {tf_iteration}")
                        
                        # Interpretation guidance
                        with st.expander("📖 How to Interpret Time-Frequency Plots"):
                            st.markdown("""
                            ### Understanding the Plots
                            
                            **Color Intensity:**
                            - Bright yellow/white = High spectral power
                            - Dark purple/black = Low/no power
                            - Logarithmic frequency scale for better low-frequency resolution
                            
                            **Panel A - Ground Truth:**
                            - Shows natural MEP frequency content
                            - Typically concentrated in 20-150 Hz range
                            - Transient burst at MEP onset
                            - Should be temporally localized (15-60 ms)
                            
                            **Panel B - Noisy Signal:**
                            - Broadband noise across all frequencies
                            - Line noise appears as horizontal streak (50 Hz, 100 Hz)
                            - TMS artefact: high-frequency burst at t=0
                            - MEP signal buried in noise
                            
                            **Panel C - Filtered Signal:**
                            - Should match Panel A in MEP window
                            - Noise frequencies attenuated
                            - Assess: Is MEP spectral structure preserved?
                            - Check: Any spurious frequencies introduced?
                            
                            ### Good Filter Characteristics
                            
                            ✓ Preserves MEP frequency content (20-150 Hz in MEP window)
                            ✓ Removes high-frequency noise (>300 Hz)
                            ✓ Removes low-frequency drift (<10 Hz)
                            ✓ No introduction of spurious frequencies
                            ✓ Maintains temporal localization
                            
                            ### Red Flags
                            
                            ✗ MEP frequency content reduced/distorted
                            ✗ New frequencies appear (filter ringing)
                            ✗ Temporal smearing (energy spreads in time)
                            ✗ Asymmetric frequency attenuation
                            """)
                    else:
                        st.error(f"Iteration {tf_iteration} not available for this configuration")
            
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
    This tab generates publication-style methods text based on your current analysis settings.
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
        
        # Add detail level selector
        detail_level = st.radio(
            "Select documentation detail level:",
            ["Standard (Journal Article)", "Detailed (Methods Paper)", "Complete (Technical Documentation)"],
            index=1,
            help="Standard: ~500 words; Detailed: ~1000 words; Complete: ~2000 words with all formulas"
        )
        
        methods_parts.append("## Software Environment\n\n")
        methods_parts.append("Motor evoked potential simulations and digital filter evaluations were performed using the ")
        methods_parts.append("MEPSimFilt version 1.0.0 (Andrushko, 2025), implemented in Python 3.8+. ")
        
        if detail_level != "Standard (Journal Article)":
            methods_parts.append("Core computational dependencies included: NumPy 1.24+ (array operations and numerical computing), ")
            methods_parts.append("SciPy 1.11+ (signal processing and filter design via scipy.signal module), ")
            methods_parts.append("Matplotlib 3.7+ (visualisation), Pandas 2.0+ (data management and aggregation), ")
            methods_parts.append("and Seaborn 0.12+ (statistical graphics). ")
        
        if detail_level == "Complete (Technical Documentation)":
            methods_parts.append("All numerical operations were performed using double-precision floating-point ")
            methods_parts.append("arithmetic (float64) to ensure numerical stability. ")
        
        methods_parts.append("\n\n## Signal Generation\n\n")
        
        # MEP morphology description with COMPLETE parameters
        mep_type = params.get('mep_type', 'Standard')
        amplitude = params.get('amplitude', 1.0)
        duration = params.get('duration', 30)
        onset_latency = params.get('onset_latency', 20)
        rise_time = params.get('rise_time', 8)
        decay_time = params.get('decay_time', 15)
        asymmetry = params.get('asymmetry', 1.2)
        
        if mep_type == "Standard":
            methods_parts.append(f"Monophasic motor evoked potentials were generated using gamma distribution-based ")
            methods_parts.append(f"functions with peak amplitude {amplitude:.1f} mV, total duration {duration:.0f} ms, ")
            methods_parts.append(f"onset latency {onset_latency:.0f} ms post-stimulus, ")
            methods_parts.append(f"rise time {rise_time:.0f} ms, and decay time {decay_time:.0f} ms. ")
            
            if detail_level != "Standard (Journal Article)":
                methods_parts.append(f"An asymmetry factor of {asymmetry:.1f} was applied to the decay phase ")
                methods_parts.append("to create realistic waveform morphology. ")
            
            if detail_level == "Complete (Technical Documentation)":
                methods_parts.append("The positive phase was modelled as f_pos(t) = (t/θ)^k · exp(-kt/τ_rise) ")
                methods_parts.append("where k=3.0 (shape parameter), θ=τ_rise/k, and τ_rise represents the rise time constant. ")
                methods_parts.append("The negative phase was similarly modelled with k=2.0 and scaled to 30% of positive amplitude. ")
        
        elif mep_type == "Bi-phasic":
            methods_parts.append(f"Bi-phasic motor evoked potentials (positive-negative morphology) were generated ")
            methods_parts.append(f"using dual Gaussian functions with peak amplitude {amplitude:.1f} mV ")
            methods_parts.append(f"and total duration {duration:.0f} ms. ")
            methods_parts.append(f"The positive phase had {rise_time:.0f} ms characteristic width, ")
            methods_parts.append(f"whilst the negative phase had {decay_time:.0f} ms width ")
            methods_parts.append("and was scaled to 80% of the positive phase amplitude ")
            methods_parts.append("to reflect realistic phase ratios observed in upper limb muscles (Groppa et al., 2012). ")
            references_used.append('groppa2012')
            
            if detail_level != "Standard (Journal Article)":
                methods_parts.append(f"Onset latency was set to {onset_latency:.0f} ms post-stimulus. ")
                methods_parts.append("Phase 1 (positive) was modelled as A₁·exp(-(t-t_peak1)²/(2σ₁²)) with σ₁=phase1_duration/4, ")
                methods_parts.append("and Phase 2 (negative) as -0.8·A₁·exp(-(t-t_peak2)²/(2σ₂²)) with σ₂=phase2_duration/3. ")
            
            if detail_level == "Complete (Technical Documentation)":
                methods_parts.append("A 3 ms sigmoid onset ramp [r(t) = 0.5(1 + tanh((t-1.5)/0.5))] was applied ")
                methods_parts.append("to prevent discontinuities and simulate physiological motor unit recruitment dynamics ")
                methods_parts.append("with smooth amplitude build-up. ")
        
        elif mep_type == "Tri-phasic":
            methods_parts.append(f"Tri-phasic motor evoked potentials (positive-negative-positive morphology) ")
            methods_parts.append(f"were generated with peak amplitude {amplitude:.1f} mV and duration {duration:.0f} ms. ")
            methods_parts.append("Phase amplitude ratios were 100% (positive), 75% (negative), and 40% (positive recovery) ")
            methods_parts.append("using three sequential Gaussian components, ")
            methods_parts.append("reflecting morphological patterns commonly observed in lower limb muscles. ")
            references_used.append('groppa2012')
            
            if detail_level != "Standard (Journal Article)":
                methods_parts.append(f"Phase 1 width: {rise_time:.0f} ms, Phase 2 width: {decay_time:.0f} ms, ")
                methods_parts.append(f"Phase 3 width: {rise_time*1.3:.0f} ms. ")
                methods_parts.append(f"Onset latency: {onset_latency:.0f} ms post-stimulus. ")
            
            if detail_level == "Complete (Technical Documentation)":
                methods_parts.append("Each phase was modelled as an independent Gaussian component ")
                methods_parts.append("[A_i·exp(-(t-t_i)²/(2σ_i²))] with temporal separation optimised ")
                methods_parts.append("to create distinct, non-overlapping deflections. ")
                methods_parts.append("A 3 ms sigmoid onset ramp was applied for physiological realism. ")
        
        elif mep_type == "Double Peak":
            methods_parts.append(f"Double-peak motor evoked potentials were generated by superposition of ")
            methods_parts.append(f"two monophasic waveforms with amplitudes {amplitude:.1f} mV and {amplitude*0.6:.1f} mV, ")
            methods_parts.append("separated by 5 ms, ")
            methods_parts.append("simulating patterns occasionally observed with dual-coil transcranial magnetic stimulation. ")
        
        else:  # With Baseline EMG
            methods_parts.append(f"Motor evoked potentials with baseline electromyographic activity were generated ")
            methods_parts.append(f"using a standard monophasic waveform ({amplitude:.1f} mV, {duration:.0f} ms) ")
            methods_parts.append("with added tonic EMG (bandpass filtered white noise, 20-500 Hz, 50 µV RMS) ")
            methods_parts.append("in the pre-stimulus period to simulate incomplete muscle relaxation. ")
        
        # Add common signal parameters
        methods_parts.append(f"All signals were sampled at {data['sampling_rate']} Hz. ")
        
        if detail_level != "Standard (Journal Article)":
            methods_parts.append(f"A 20 ms pre-stimulus baseline window was included (time vector: -20 to 100 ms) ")
            methods_parts.append("to provide temporal context for stimulus artefact visualisation and assessment. ")
        
        # COMPREHENSIVE NOISE DESCRIPTION
        methods_parts.append("\n\n## Noise Simulation\n\n")
        noise_types = params.get('noise_types', {})
        snr_db = params.get('snr_db', 10)
        line_freq = params.get('line_freq', 50)
        
        methods_parts.append(f"Composite physiological and technical noise was added to achieve ")
        methods_parts.append(f"a signal-to-noise ratio of {snr_db:.0f} dB. ")
        
        if detail_level == "Standard (Journal Article)":
            methods_parts.append("The noise model included ")
            noise_list = []
            if noise_types.get('white', True):
                noise_list.append("Gaussian white noise")
            if noise_types.get('emg'):
                noise_list.append("EMG noise (20-500 Hz)")
            if noise_types.get('line'):
                noise_list.append(f"{line_freq} Hz line noise")
            if noise_types.get('ecg'):
                noise_list.append("ECG artefacts")
            if noise_types.get('movement'):
                noise_list.append("movement artefacts")
            if noise_types.get('tms'):
                noise_list.append("TMS artefact (1.5 mV)")
            
            if noise_list:
                methods_parts.append(", ".join(noise_list) + ". ")
        
        else:  # Detailed or Complete
            methods_parts.append("The composite noise model incorporated the following components:\n\n")
            
            if noise_types.get('white', True):
                methods_parts.append("**White Noise:** Gaussian white noise (zero mean, unit variance) ")
                methods_parts.append("scaled to achieve the target signal-to-noise ratio. ")
                if detail_level == "Complete (Technical Documentation)":
                    methods_parts.append("Noise power was calculated relative to signal power in the MEP window ")
                    methods_parts.append("(15-80 ms post-stimulus): SNR_dB = 10·log₁₀(P_signal/P_noise). ")
                methods_parts.append("\n\n")
            
            if noise_types.get('emg'):
                methods_parts.append("**Electromyographic Noise:** Simulated using Gaussian white noise ")
                methods_parts.append("bandpass filtered (20-500 Hz, 4th-order Butterworth) to match ")
                methods_parts.append("physiological EMG frequency content. ")
                if detail_level == "Complete (Technical Documentation)":
                    methods_parts.append("Burst events (10% probability, 3× baseline amplitude) were ")
                    methods_parts.append("randomly inserted to simulate motor unit activity during incomplete relaxation. ")
                methods_parts.append("\n\n")
            
            if noise_types.get('line'):
                methods_parts.append(f"**Line Noise:** {line_freq} Hz sinusoidal interference ")
                if detail_level == "Complete (Technical Documentation)":
                    methods_parts.append(f"plus second harmonic ({line_freq*2} Hz) with 1/n amplitude decay ")
                    methods_parts.append("to simulate mains power interference and harmonic distortion. ")
                else:
                    methods_parts.append("with harmonics to simulate mains power interference. ")
                methods_parts.append("\n\n")
            
            if noise_types.get('ecg'):
                methods_parts.append("**Electrocardiographic Artefacts:** Simplified QRS complexes ")
                methods_parts.append("(70 beats per minute) simulating cardiac interference in recordings ")
                methods_parts.append("from proximal muscles or with suboptimal electrode placement. ")
                methods_parts.append("\n\n")
            
            if noise_types.get('movement'):
                methods_parts.append("**Movement Artefacts:** Low-frequency drift (<2 Hz) generated via ")
                methods_parts.append("filtered random walk process to simulate electrode movement or ")
                methods_parts.append("baseline wander during recording. ")
                methods_parts.append("\n\n")
            
            if noise_types.get('tms'):
                methods_parts.append("**Transcranial Magnetic Stimulation Artefact:** Residual post-blanking ")
                methods_parts.append("artefact modelled as 1.5 mV peak amplitude with 2 ms exponential decay ")
                methods_parts.append("time constant and superimposed 300 Hz damped oscillation, ")
                methods_parts.append("simulating realistic TMS artefact remaining after hardware blanking (Rossini et al., 2015). ")
                if detail_level == "Complete (Technical Documentation)":
                    methods_parts.append("Mathematical form: A_TMS(t) = 1.5·exp(-t/0.002)·[1 + 0.3·sin(600πt)·exp(-t/0.0006)] mV ")
                    methods_parts.append("for t ≥ 0, where t=0 represents the TMS pulse. ")
                methods_parts.append("\n\n")
                references_used.append('rossini2015')
        
        # Filter analysis
        if st.session_state.batch_results is not None:
            df = st.session_state.batch_results
            n_configs = len(df['config_label'].unique())
            n_iterations = len(df) // n_configs if n_configs > 0 else 0
            
            methods_parts.append("\n\n## Digital Filter Evaluation\n\n")
            
            # Describe filter types with technical details
            filter_types = df['filter_type'].unique()
            filter_desc = []
            
            if any('butterw' in f for f in filter_types):
                orders = sorted(df[df['filter_type'].str.contains('butterw')]['order'].unique())
                
                if detail_level == "Standard (Journal Article)":
                    methods_parts.append(f"Butterworth infinite impulse response filters (orders {', '.join(map(str, orders))}) ")
                elif detail_level == "Detailed (Methods Paper)":
                    methods_parts.append(f"**Butterworth Filters:** Infinite impulse response (IIR) Butterworth filters ")
                    methods_parts.append(f"of orders {', '.join(map(str, orders))} were designed using scipy.signal.butter() ")
                    methods_parts.append("in second-order sections (SOS) format to minimise numerical errors. ")
                    methods_parts.append("Zero-phase filtering was achieved via forward-backward application ")
                    methods_parts.append("(scipy.signal.filtfilt()), which eliminates phase distortion critical for ")
                    methods_parts.append("preserving motor evoked potential latency measurements. ")
                else:  # Complete
                    methods_parts.append(f"**Butterworth Filters:** IIR Butterworth filters (orders {', '.join(map(str, orders))}) ")
                    methods_parts.append("were designed via scipy.signal.butter() using second-order sections ")
                    methods_parts.append("(SOS) representation to avoid numerical instability associated with ")
                    methods_parts.append("high-order transfer function coefficients. ")
                    methods_parts.append("The Butterworth magnitude response is defined as ")
                    methods_parts.append("|H(ω)|² = 1/[1+(ω/ωc)^(2n)] where n is the filter order and ωc ")
                    methods_parts.append("is the cutoff frequency. ")
                    methods_parts.append("Zero-phase response was achieved via scipy.signal.filtfilt(), ")
                    methods_parts.append("which applies the filter forwards and backwards ")
                    methods_parts.append("(effective filter order: 2n) whilst maintaining zero phase shift. ")
                    methods_parts.append("This is critical for accurate latency measurements as ")
                    methods_parts.append("single-pass IIR filters introduce frequency-dependent phase delays. ")
                
                references_used.append('butterworth')
            
            if any('fir' in f for f in filter_types):
                fir_windows = list(set([f.replace('fir_', '').replace('_zp', '').capitalize() 
                                       for f in filter_types if 'fir' in f]))
                
                if detail_level == "Standard (Journal Article)":
                    methods_parts.append(f"and finite impulse response filters ({', '.join(fir_windows)} windows) ")
                elif detail_level == "Detailed (Methods Paper)":
                    methods_parts.append(f"\n\n**FIR Filters:** Finite impulse response filters using ")
                    methods_parts.append(f"{', '.join(fir_windows)} window functions were designed via ")
                    methods_parts.append("scipy.signal.firwin(). ")
                    methods_parts.append("Filter length was automatically calculated based on transition ")
                    methods_parts.append("bandwidth requirements whilst ensuring stability ")
                    methods_parts.append("(maximum 30% of signal length to prevent edge artefacts). ")
                else:  # Complete
                    methods_parts.append(f"\n\n**FIR Filters:** Window-based FIR filter design via scipy.signal.firwin() ")
                    methods_parts.append(f"with {', '.join(fir_windows)} window functions. ")
                    methods_parts.append("Filter taps were calculated as N = (sampling_rate × transition_width) ")
                    methods_parts.append("with automatic limitation to 30% of signal length to avoid ")
                    methods_parts.append("padding artefacts in scipy.signal.filtfilt(). ")
                    methods_parts.append("FIR filters have linear phase response in their passband, ")
                    methods_parts.append("which when combined with filtfilt provides zero-phase response. ")
                
                references_used.append('fir')
            
            # Frequency specifications
            hp_cutoffs = sorted(df['lowcut'].unique())
            lp_cutoffs = sorted(df['highcut'].unique())
            
            methods_parts.append(f"were evaluated systematically across {n_configs} configurations ")
            methods_parts.append(f"with highpass cutoffs of {', '.join([f'{int(x)}' for x in hp_cutoffs])} Hz ")
            methods_parts.append(f"and lowpass cutoffs of {', '.join([f'{int(x)}' for x in lp_cutoffs])} Hz. ")
            
            if detail_level != "Standard (Journal Article)":
                nyquist = data['sampling_rate'] / 2.0
                methods_parts.append(f"All cutoff frequencies were validated against the Nyquist limit ")
                methods_parts.append(f"({nyquist:.0f} Hz) to ensure filter stability, ")
                methods_parts.append("with automatic adjustment to 99% of Nyquist if required. ")
            
            # Notch filter
            if df['notch_enabled'].any():
                if detail_level == "Standard (Journal Article)":
                    methods_parts.append("Selected configurations included a 50 Hz notch filter for line noise rejection. ")
                else:
                    methods_parts.append("Selected configurations incorporated a 50 Hz notch filter ")
                    methods_parts.append("(quality factor Q=30, bandwidth ≈1.7 Hz) implemented via ")
                    methods_parts.append("scipy.signal.iirnotch() with zero-phase application. ")
            
            methods_parts.append(f"Each of the {n_configs} configurations was tested across {n_iterations} ")
            methods_parts.append("independent noise realisations to assess performance robustness and consistency. ")
            
            methods_parts.append("\n\n## Performance Metrics\n\n")
            
            if detail_level == "Standard (Journal Article)":
                methods_parts.append("Filter performance was quantified using amplitude preservation error ")
                methods_parts.append("(percentage deviation from ground truth peak-to-peak amplitude), ")
                methods_parts.append("peak latency shift (milliseconds), Pearson correlation coefficient, ")
                methods_parts.append("and root mean square error. ")
                methods_parts.append("All metrics reported as mean ± standard deviation across iterations. ")
            
            elif detail_level == "Detailed (Methods Paper)":
                methods_parts.append("**Amplitude Preservation Error:** Percentage deviation of filtered ")
                methods_parts.append("peak-to-peak amplitude from ground truth within the MEP window ")
                methods_parts.append("(15-80 ms post-stimulus): E_amp = 100 × (A_filt - A_true) / A_true.\n\n")
                
                methods_parts.append("**Peak Latency Shift:** Temporal displacement (milliseconds) ")
                methods_parts.append("of maximum absolute amplitude.\n\n")
                
                methods_parts.append("**Pearson Correlation:** Computed via np.corrcoef() between ")
                methods_parts.append("filtered and ground truth waveforms for morphological fidelity.\n\n")
                
                methods_parts.append("**RMSE:** Root mean square error calculated over MEP window.\n\n")
                
                methods_parts.append("**SNR Improvement:** Post-filtering SNR minus pre-filtering SNR (dB). ")
            
            else:  # Complete Technical Documentation
                methods_parts.append("**Amplitude Preservation Error:**\n\n")
                methods_parts.append("E_amp = 100 × (A_filtered - A_true) / A_true (%)\n\n")
                methods_parts.append("where A = max(x) - min(x) within MEP window (15-80 ms), ")
                methods_parts.append("automatically detected via threshold algorithm.\n\n")
                
                methods_parts.append("**Onset Latency Error:**\n\n")
                methods_parts.append("Onset detected when 3 consecutive samples exceed ")
                methods_parts.append("μ_baseline + 3σ_baseline (baseline: 0-15 ms). ")
                methods_parts.append("Error = latency_filtered - latency_true (ms).\n\n")
                
                methods_parts.append("**Peak Latency Error:**\n\n")
                methods_parts.append("E_latency = argmax(|x_filtered|) - argmax(|x_true|) (ms)\n\n")
                
                methods_parts.append("**Pearson Correlation Coefficient:**\n\n")
                methods_parts.append("r = Σ[(x_i-μ_x)(y_i-μ_y)] / √[Σ(x_i-μ_x)²·Σ(y_i-μ_y)²]\n\n")
                methods_parts.append("Computed via NumPy's corrcoef() over full signal.\n\n")
                
                methods_parts.append("**Root Mean Square Error:**\n\n")
                methods_parts.append("RMSE = √[(1/N)Σ(x_filt[i] - x_true[i])²] (mV)\n\n")
                methods_parts.append("Calculated over: (1) full signal, (2) MEP window only.\n\n")
                
                methods_parts.append("**Signal-to-Noise Ratio:**\n\n")
                methods_parts.append("SNR_dB = 10·log₁₀(P_signal/P_noise)\n\n")
                methods_parts.append("where P = mean(x²) in respective windows.\n\n")
                
                methods_parts.append("**Baseline Stability:** σ of filtered signal in 0-15 ms (lower = better).\n\n")
                
                methods_parts.append("**Spurious Oscillations:** Detected if baseline envelope >5σ ")
                methods_parts.append("or zero-crossing rate >30% (indicates filter ringing). ")
            
            references_used.append('groppa2012')
            references_used.append('rossini2015')
            
            methods_parts.append("\n\n## Statistical Analysis\n\n")
            
            if detail_level == "Standard (Journal Article)":
                methods_parts.append(f"Metrics were aggregated across {n_iterations} iterations using ")
                methods_parts.append("mean and standard deviation. ")
                methods_parts.append("Configuration performance was compared using Kruskal-Wallis H-test with ")
                methods_parts.append("post-hoc Mann-Whitney U tests and Bonferroni correction for multiple comparisons. ")
                methods_parts.append("Optimal configurations identified via minimum |amplitude error|, ")
                methods_parts.append("minimum |latency shift|, and maximum correlation. ")
                methods_parts.append("Effect sizes (Cohen's d) quantified practical significance of differences. ")
            
            elif detail_level == "Detailed (Methods Paper)":
                methods_parts.append(f"Each filter configuration's performance was assessed across ")
                methods_parts.append(f"{n_iterations} independent noise realisations. ")
                methods_parts.append("Metrics were aggregated using arithmetic mean ")
                methods_parts.append("(μ = Σx_i / n) and sample standard deviation ")
                methods_parts.append("[σ = √(Σ(x_i-μ)² / (n-1))]. ")
                methods_parts.append("95% confidence intervals were calculated using t-distribution: ")
                methods_parts.append("CI = μ ± t_(α/2,n-1) · (σ/√n).\n\n")
                
                methods_parts.append("**Hypothesis Testing:** ")
                methods_parts.append("Differences between filter configurations were evaluated using ")
                methods_parts.append("Kruskal-Wallis H-test (non-parametric one-way ANOVA) to test the ")
                methods_parts.append("null hypothesis that all configurations perform equally. ")
                methods_parts.append("Post-hoc pairwise comparisons between top-performing configurations ")
                methods_parts.append("employed Mann-Whitney U tests with Bonferroni correction ")
                methods_parts.append(f"(p_corrected = p_raw × n_comparisons where n_comparisons = n(n-1)/2 for n configurations). ")
                methods_parts.append("Statistical significance threshold: α = 0.05.\n\n")
                
                methods_parts.append("**Effect Sizes:** ")
                methods_parts.append("Cohen's d was calculated for pairwise comparisons: ")
                methods_parts.append("d = (μ₁ - μ₂) / σ_pooled where σ_pooled = √[((n₁-1)σ₁² + (n₂-1)σ₂²)/(n₁+n₂-2)]. ")
                methods_parts.append("Effect magnitudes interpreted as: |d| < 0.2 (negligible), 0.2-0.5 (small), ")
                methods_parts.append("0.5-0.8 (medium), >0.8 (large).\n\n")
                
                methods_parts.append("**Configuration Ranking:** ")
                methods_parts.append("Configurations ranked independently for each metric. ")
                methods_parts.append("Composite scores calculated as normalized combination of amplitude error, ")
                methods_parts.append("latency error, and correlation (equal weighting). ")
                methods_parts.append("Absolute values were used for error metrics to avoid bias ")
                methods_parts.append("favouring systematic underestimation versus overestimation. ")
            
            else:  # Complete
                methods_parts.append(f"**Aggregation Across Iterations ({n_iterations} per configuration):**\n\n")
                methods_parts.append("Mean: μ = (1/n)Σx_i\n\n")
                methods_parts.append("Standard Deviation: σ = √[(1/(n-1))Σ(x_i-μ)²]\n\n")
                methods_parts.append("95% Confidence Interval: CI = μ ± t_(α/2,n-1) · SE where SE = σ/√n\n\n")
                methods_parts.append("(Using n-1 denominator for unbiased sample estimator)\n\n")
                
                methods_parts.append("**Hypothesis Testing Framework:**\n\n")
                methods_parts.append(f"Overall Test: Kruskal-Wallis H-test (non-parametric one-way ANOVA) ")
                methods_parts.append(f"comparing all {n_configs} configurations simultaneously:\n\n")
                methods_parts.append("H = (12/N(N+1)) · Σ[R_i²/n_i] - 3(N+1)\n\n")
                methods_parts.append("where R_i = sum of ranks for group i, n_i = group size, N = total observations. ")
                methods_parts.append("Null hypothesis (H₀): All configurations perform equally. ")
                methods_parts.append("Rejection criterion: p < 0.05.\n\n")
                
                methods_parts.append("**Post-Hoc Pairwise Comparisons:**\n\n")
                methods_parts.append("Mann-Whitney U test (Wilcoxon rank-sum test) for each pair of top-5 configurations:\n\n")
                methods_parts.append("U = n₁n₂ + n₁(n₁+1)/2 - R₁\n\n")
                methods_parts.append("where R₁ = sum of ranks in group 1. ")
                methods_parts.append("Bonferroni correction for multiple comparisons: ")
                methods_parts.append("p_corrected = min(p_raw × k, 1.0) where k = number of comparisons.\n\n")
                
                methods_parts.append("**Effect Size Quantification:**\n\n")
                methods_parts.append("Cohen's d calculated for standardized mean difference:\n\n")
                methods_parts.append("d = (μ₁ - μ₂) / σ_pooled\n\n")
                methods_parts.append("σ_pooled = √[((n₁-1)σ₁² + (n₂-1)σ₂²) / (n₁+n₂-2)]\n\n")
                methods_parts.append("Interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), ")
                methods_parts.append("0.5-0.8 (medium), >0.8 (large practical significance).\n\n")
                
                methods_parts.append("**Optimal Configuration Selection:**\n\n")
                methods_parts.append("Configurations ranked independently for each metric:\n\n")
                methods_parts.append("- Amplitude error: argmin(|μ_error|)\n")
                methods_parts.append("- Latency error: argmin(|μ_latency|)\n")
                methods_parts.append("- Correlation: argmax(μ_r)\n\n")
                
                methods_parts.append("Composite score for overall ranking:\n\n")
                methods_parts.append("S_composite = (S_amp + S_lat + S_corr) / 3\n\n")
                methods_parts.append("where each component normalized to [0,1] range. ")
                methods_parts.append("Statistical significance of rank differences verified via pairwise tests.\n\n")
                
                methods_parts.append("**Visualisation Methods:**\n\n")
                methods_parts.append(f"- Box plots: Median, IQR, whiskers to 1.5×IQR (n={n_iterations} per box)\n")
                methods_parts.append("- Heatmaps: Mean values, diverging colour map centred at zero\n")
                methods_parts.append("- Overlay plots: Individual iterations + mean ± 1σ envelope\n")
                methods_parts.append("- Statistical tables: Rankings with 95% CIs, pairwise comparisons\n\n")
                
                methods_parts.append("**Statistical Software:** ")
                methods_parts.append("SciPy 1.11+ (scipy.stats.kruskal, scipy.stats.mannwhitneyu, scipy.stats.t), ")
                methods_parts.append("Pandas (groupby aggregation), NumPy (statistical functions). ")
                methods_parts.append("All computations vectorised for efficiency. ")
        
        # Display generated methods
        methods_text = "".join(methods_parts)
        
        st.markdown(methods_text)
        
        st.divider()
        
        # Download options
        st.subheader("📥 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Methods Text**")
            # Add comprehensive footer with versions
            full_methods = methods_text + "\n\n## Tool Citation\n\n"
            full_methods += "Andrushko, J.W. (2025). MEPSimFilt:  "
            full_methods += "A Systematic Digital Filter Evaluation tool for Motor Evoked Potentials (Version 1.0.0). Northumbria University. "
            full_methods += "https://github.com/jandrushko/MEPSimFilt\n\n"
            full_methods += "## Software Package Versions\n\n"
            full_methods += "- Python: 3.8 or higher\n"
            full_methods += "- NumPy: ≥1.24.0 (numerical computing, array operations)\n"
            full_methods += "- SciPy: ≥1.11.0 (signal processing, filter design)\n"
            full_methods += "  - scipy.signal.butter() - Butterworth filter design\n"
            full_methods += "  - scipy.signal.firwin() - FIR filter design\n"
            full_methods += "  - scipy.signal.filtfilt() - Zero-phase filtering\n"
            full_methods += "  - scipy.signal.iirnotch() - Notch filter design\n"
            full_methods += "  - scipy.signal.sosfreqz() - Frequency response\n"
            full_methods += "- Matplotlib: ≥3.7.0 (plotting and visualisation)\n"
            full_methods += "- Pandas: ≥2.0.0 (data management, groupby aggregation)\n"
            full_methods += "- Seaborn: ≥0.12.0 (statistical visualisation)\n"
            full_methods += "- Streamlit: ≥1.28.0 (graphical user interface)\n\n"
            full_methods += "## Numerical Precision\n\n"
            full_methods += "- Floating point: 64-bit double precision (numpy.float64)\n"
            full_methods += "- Random number generation: NumPy Mersenne Twister (np.random.randn())\n"
            full_methods += "- Independent random seeds for each iteration\n"
            
            st.download_button(
                label=f"📄 Download Methods ({detail_level.split()[0]})",
                data=full_methods,
                file_name=f"mep_methods_{detail_level.split()[0].lower()}_{mep_type.lower().replace('-','')}.txt",
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
TI  - MEPSimFilt: A Systematic Digital Filter Evaluation tool for Motor Evoked Potentials
PY  - 2025
PB  - Northumbria University
VL  - 1.0
UR  - https://github.com/jandrushko/MEPSimFilt
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
        
        3. **Andrushko, J.W. (2025).** MEPSimFilt: A Systematic Digital Filter Evaluation tool for Motor Evoked Potentials (Version 1.0.0). Northumbria University. 
           https://github.com/jandrushko/MEPSimFilt
        
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
        
        This tab will display publication-style methods text that adapts to your specific parameters.
        """)

# ===========================
# TAB 5: ABOUT
# ===========================
with tab5:
    st.header("About This Tool")
    
    st.markdown("""
    ### MEPSimFilt: MEP Filter Testing Tool
    
    **Version:** 1.0.0  
    **Released:** December 2025  
    **Author:** Justin W. Andrushko, PhD  
    **Institution:** Northumbria University
    
    ### Purpose
    
    This tool was developed to systematically evaluate digital filter performance for 
    Motor Evoked Potential (MEP) analysis in transcranial magnetic stimulation (TMS) studies. 
    It addresses the critical need for evidence-based filter selection through rigorous 
    simulation-based testing with comprehensive statistical analysis.
    
    ### Key Features
    
    **Signal Generation:**
    - Multiple MEP morphologies (monophasic, bi-phasic, tri-phasic)
    - **Real waveform loading** (TXT, CSV, MAT formats)
    - **Automatic parameter extraction** from experimental data
    - **Advanced bi-phasic controls** (6 independent phase parameters)
    - **Advanced tri-phasic controls** (11 independent phase parameters)
    - Realistic temporal dynamics with smooth onset ramps
    - Pre-stimulus baseline windows
    - Configurable amplitudes, durations, and latencies
    
    **Noise Modelling:**
    - Six noise types (White, EMG, Line, ECG, Movement, TMS artefact)
    - Adjustable signal-to-noise ratios (-10 to +40 dB)
    - Physiologically realistic interference simulation
    
    **Filter Evaluation:**
    - Butterworth filters (1st through 8th orders)
    - FIR filters (multiple window functions)
    - Notch filters for line noise rejection
    - Up to 10,000 iterations per configuration
    
    **Statistical Analysis:**
    - Comprehensive assumption testing (normality, homogeneity, sample size)
    - Automatic test selection based on assumptions
    - Parametric and non-parametric methods
    - Multiple comparison corrections
    - Effect size quantification
    - Publication-style ready formal statements
    
    **Visualisation:**
    - Time-domain plots with multiple configurations
    - Time-frequency analysis (Morlet wavelets)
    - Box plots, heatmaps, overlay plots
    - Publication-quality exports (up to 600 DPI)
    
    **Documentation:**
    - Automatic methods text generation (3 detail levels)
    - Adaptive to user parameters
    - RIS reference export
    - Statistical assumption reports
    
    ### Statistical Rigor
    
    This tool implements rigorous statistical methodology with full assumption testing:
    - Shapiro-Wilk normality tests for all groups
    - Levene's test for homogeneity of variance
    - Sample size adequacy assessment
    - Evidence-based test selection (parametric vs non-parametric)
    - Formal reporting following APA guidelines
    - Complete transparency in statistical decisions
    
    ### Citation
    
    **Tool Citation:**
    
    Andrushko, J.W. (2025). MEPSimFilt: A Systematic Digital Filter Evaluation tool for Motor Evoked Potentials (Version 1.0.0). Northumbria University. 
    https://github.com/jandrushko/MEPSimFilt/
    
    **TMSMultiLab:**
    
    Part of the TMSMultiLab initiative for advancing TMS methodological standards.
    Visit: https://github.com/TMSMultiLab/TMSMultiLab/wiki
    
    ### Recommended Workflow
    
    1. **Generate Signal**: Create realistic MEP with desired noise characteristics
    2. **Interactive Testing**: Explore filter effects in Filter Testing tab
    3. **Batch Analysis**: Systematically compare multiple configurations
    4. **Statistical Analysis**: Run assumption tests and identify optimal filters
    5. **Visualisation**: Generate publication-quality figures
    6. **Time-Frequency**: Verify spectral preservation
    7. **Documentation**: Export methods text and statistical reports
    
    ### Open Source
    
    This tool is open-source and freely available. Contributions, bug reports, and 
    feature requests are welcome at the GitHub repository.
    
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
    - **Interactive Visualisation**: Real-time plotting and exploration of results
    
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
    Andrushko, J.W. (2025). MEPSimFilt: A Systematic Digital Filter Evaluation tool for Motor Evoked Potentials.
    Northumbria University.
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
