"""
EPSimFilt - Streamlit GUI
Interactive tool for testing and comparing digital filters for EP analysis.
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

# CRITICAL: Force reload of modules to avoid cache issues
import importlib
import signal_generator
import noise_generator
import filters as filters_module
import metrics as metrics_module

# Reload modules to get latest changes
importlib.reload(signal_generator)
importlib.reload(noise_generator)
importlib.reload(filters_module)
importlib.reload(metrics_module)

from signal_generator import EPGenerator
from noise_generator import NoiseGenerator
from filters import EPFilters
from metrics import EPMetrics
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
        
        # Determine EP channel (highest variability, excluding trigger channels)
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
            # Select channel with highest range (likely EP)
            EP_ch_idx = max(channel_data, key=lambda x: x[2])[0]
        else:
            EP_ch_idx = 1  # Default
        
        signal = data_array[:, EP_ch_idx]
        metadata['selected_channel'] = EP_ch_idx
        metadata['selected_channel_name'] = channel_titles[EP_ch_idx - 1] if EP_ch_idx - 1 < len(channel_titles) else f"Channel {EP_ch_idx}"
        
        # Detect trigger channel
        for ch_idx in range(1, data_array.shape[1]):
            if ch_idx == EP_ch_idx:
                continue  # Skip EP channel
            
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
            # Common EP duration is 50-150 ms
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
        for var_name in ['data', 'signal', 'emg', 'EP', 'waveform', 'y', 'amplitude']:
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


def extract_EP_parameters(time, signal, sampling_rate):
    """
    Automatically extract EP parameters from real waveform.
    
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
    
    # Focus on post-stimulus period for EP detection
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
    
    # Find EP onset (first point exceeding 10% of peak amplitude)
    threshold = params['peak_amplitude'] * 0.1
    above_threshold = np.where(np.abs(post_signal) > threshold)[0]
    
    if len(above_threshold) > 0:
        onset_idx_post = above_threshold[0]
        onset_idx = post_stim_idx[onset_idx_post]
        params['EP_onset_ms'] = float(time_ms[onset_idx])
        params['onset_latency_ms'] = float(time_ms[onset_idx] - time_ms[stim_idx])
        params['onset_idx'] = int(onset_idx)  # Store for plotting
    else:
        params['EP_onset_ms'] = params['peak_latency_ms'] - 10
        params['onset_latency_ms'] = params['EP_onset_ms']
        params['onset_idx'] = max(0, peak_idx - int(0.010 * sampling_rate))
    
    # Find EP offset (return to <10% threshold after peak)
    after_peak = post_signal[peak_idx_post:]
    below_threshold = np.where(np.abs(after_peak) < threshold)[0]
    
    if len(below_threshold) > 0:
        offset_idx_post = peak_idx_post + below_threshold[0]
        if offset_idx_post < len(post_stim_idx):
            offset_idx = post_stim_idx[offset_idx_post]
            params['EP_offset_ms'] = float(time_ms[offset_idx])
            params['offset_idx'] = int(offset_idx)  # Store for plotting
        else:
            params['EP_offset_ms'] = params['peak_latency_ms'] + 30
            params['offset_idx'] = min(len(signal_corrected) - 1, peak_idx + int(0.030 * sampling_rate))
    else:
        params['EP_offset_ms'] = params['peak_latency_ms'] + 30
        params['offset_idx'] = min(len(signal_corrected) - 1, peak_idx + int(0.030 * sampling_rate))
    
    # Calculate durations
    params['duration_ms'] = params['EP_offset_ms'] - params['EP_onset_ms']
    params['rise_time_ms'] = params['peak_latency_ms'] - params['EP_onset_ms']
    params['decay_time_ms'] = params['EP_offset_ms'] - params['peak_latency_ms']
    
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
                   alpha=0.8, label=f"EP Onset ({params['EP_onset_ms']:.1f} ms)", zorder=5)
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
                   alpha=0.8, label=f"EP Offset ({params['EP_offset_ms']:.1f} ms)", zorder=5)
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


def extract_EP_parameters(time, signal, sampling_rate):
    """
    Automatically extract EP parameters from real waveform.
    
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
    
    # Focus on post-stimulus period for EP detection
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
    
    # Find EP onset (first point exceeding 10% of peak amplitude)
    threshold = params['peak_amplitude'] * 0.1
    above_threshold = np.where(np.abs(post_signal) > threshold)[0]
    
    if len(above_threshold) > 0:
        onset_idx_post = above_threshold[0]
        onset_idx = post_stim_idx[onset_idx_post]
        params['EP_onset_ms'] = float(time_ms[onset_idx])
        params['onset_latency_ms'] = float(time_ms[onset_idx] - time_ms[stim_idx])
    else:
        params['EP_onset_ms'] = params['peak_latency_ms'] - 10
        params['onset_latency_ms'] = params['EP_onset_ms']
    
    # Find EP offset (return to <10% threshold after peak)
    after_peak = post_signal[peak_idx_post:]
    below_threshold = np.where(np.abs(after_peak) < threshold)[0]
    
    if len(below_threshold) > 0:
        offset_idx_post = peak_idx_post + below_threshold[0]
        if offset_idx_post < len(post_stim_idx):
            offset_idx = post_stim_idx[offset_idx_post]
            params['EP_offset_ms'] = float(time_ms[offset_idx])
        else:
            params['EP_offset_ms'] = params['peak_latency_ms'] + 30
    else:
        params['EP_offset_ms'] = params['peak_latency_ms'] + 30
    
    # Calculate durations
    params['duration_ms'] = params['EP_offset_ms'] - params['EP_onset_ms']
    params['rise_time_ms'] = params['peak_latency_ms'] - params['EP_onset_ms']
    params['decay_time_ms'] = params['EP_offset_ms'] - params['peak_latency_ms']
    
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


def plot_timefreq_comparison(EP_clean, EP_noisy, EP_filtered, time, sampling_rate,
                             freq_range=(5, 500), n_freqs=80, vmax_percentile=98):
    """
    Create multi-panel time-frequency comparison figure.
    
    Parameters:
    -----------
    EP_clean : np.ndarray
        Ground truth EP
    EP_noisy : np.ndarray
        Noisy EP  
    EP_filtered : np.ndarray
        Filtered EP
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
    freqs_clean, tfr_clean = compute_morlet_timefreq(EP_clean, sampling_rate, freq_range, n_freqs)
    freqs_noisy, tfr_noisy = compute_morlet_timefreq(EP_noisy, sampling_rate, freq_range, n_freqs)
    freqs_filt, tfr_filt = compute_morlet_timefreq(EP_filtered, sampling_rate, freq_range, n_freqs)
    
    # Ensure all have same length (trim to shortest if needed)
    min_len = min(tfr_clean.shape[1], tfr_noisy.shape[1], tfr_filt.shape[1], len(time))
    tfr_clean = tfr_clean[:, :min_len]
    tfr_noisy = tfr_noisy[:, :min_len]
    tfr_filt = tfr_filt[:, :min_len]
    time_plot = time[:min_len]
    EP_clean_plot = EP_clean[:min_len]
    EP_noisy_plot = EP_noisy[:min_len]
    EP_filtered_plot = EP_filtered[:min_len]
    
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
    ax1_tf.set_title('A) Ground Truth EP - Time-Frequency Representation', 
                     fontsize=13, fontweight='bold', loc='left')
    ax1_tf.set_yscale('log')
    ax1_tf.set_ylim(freq_range)
    ax1_tf.grid(False)
    ax1_tf.tick_params(labelbottom=False)
    
    # Waveform
    ax1_wave.plot(time_ms, EP_clean_plot, 'k-', linewidth=2)
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
    ax2_wave.plot(time_ms, EP_noisy_plot, 'r-', linewidth=1.2, alpha=0.8)
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
    ax3_wave.plot(time_ms, EP_filtered_plot, 'b-', linewidth=2)
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


# ===========================
# STREAMLIT APP
# ===========================

# Page configuration
st.set_page_config(
    page_title="EPSimFilt",
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
    st.markdown("### EPSimFilt")
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
    Andrushko, J.W. (2025). EPSimFilt: A Systematic Digital Filter Evaluation tool
    for Evoked Potentials (Version 1.0.0). Northumbria University.
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'generated_signals' not in st.session_state:
    st.session_state.generated_signals = None
if 'filter_results' not in st.session_state:
    st.session_state.filter_results = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None


# Computation Mode Selector
with st.expander("⚙️ Computation Mode", expanded=False):
    computation_mode = st.radio(
        "Select Mode:",
        ["🎯 Demo Mode (10-50 iterations)", "🚀 Full Mode (100-10,000 iterations)"],
        index=0
    )
    is_demo_mode = "Demo" in computation_mode
    default_batch_iterations = 10 if is_demo_mode else 100
    max_batch_iterations = 50 if is_demo_mode else 10000

st.divider()

# Title and description
st.title("🧠 EPSimFilt")

st.markdown("""
**Systematically evaluate digital filter performance for Evoked Potentials (EPs)**

This tool allows you to:
- Generate realistic EP signals with configurable parameters
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
    
    # ========== REPRODUCIBILITY UPLOAD SECTION ==========
    with st.expander("🔢 Load Reproducibility Parameters (Optional)", expanded=False):
        st.markdown("""
        **Upload a reproducibility JSON file** from a previous analysis to recreate 
        the exact same EPs and settings.
        
        This file contains the precise amplitude, latency, and duration values 
        for each iteration, ensuring bit-for-bit identical reproduction.
        """)
        
        repro_file = st.file_uploader("Upload reproducibility JSON", type=["json"], key="repro_upload")
        
        if repro_file is not None:
            import json
            repro_data = json.load(repro_file)
            
            st.success("✅ Reproducibility file loaded!")
            
            # Display key info
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Total Iterations", len(repro_data.get('iteration_parameters', [])))
                st.metric("Master Seed", repro_data.get('master_seed', 'Not set'))
            with col_r2:
                st.metric("EP Type", repro_data['signal_parameters'].get('EP_type', 'N/A'))
                st.metric("Variability", "Enabled" if repro_data['signal_parameters'].get('variability_enabled') else "Disabled")
            
            # Store in session state
            st.session_state.loaded_repro_params = repro_data
            
            st.info("📋 These exact iteration parameters will be used in batch analysis")
    
    st.divider()
    
    # ========== FILE UPLOAD SECTION ==========
    st.subheader("📂 Load Real EP Waveform (Optional)")
    
    with st.expander("Upload and Auto-Extract Parameters from Real Data", expanded=False):
        st.markdown("""
        Upload a real EP waveform to automatically extract parameters and match your experimental data.
        
        **Supported formats:**
        - **LabView** (TXT: 2-row transposed format with time/data rows)
        - **LabChart** (TXT exports with headers: Interval, ChannelTitle, Range)
        - **Spike2** (TXT exports with INFORMATION, SUMMARY, START markers)
          - **Automatically detects DigMarks/Events** (Digitimer, Magstim triggers)
          - **Segments recording** around each event for individual trial analysis
        - **Multi-trial CSV** (rows = tiEPoints, columns = trials)
        - **Standard CSV** (two columns: time, amplitude)
        - **Standard TXT** (space or tab delimited)
        - **MATLAB MAT** (variables: signal/data/EP and time/t)
        
        The tool will auto-detect format and events. For Spike2 files with DigMarks, you can select specific trials!
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file containing EP waveform",
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
                        extracted_params = extract_EP_parameters(time_ext, signal_ext, fs_ext)
                        
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
                            title=f"Extracted Parameters - {params['morphology']} EP"
                        )
                        
                        st.pyplot(fig_annotated)
                        plt.close()
                        
                        st.caption("""
                        **Legend:** 
                        - 🟠 Orange dashed line = Stimulus (t=0)
                        - 🟢 Green markers = EP Onset (10% threshold crossing)
                        - 🔴 Red markers = Peak amplitude
                        - 🟣 Purple markers = EP Offset (return to <10%)
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
                            loaded_include_line = st.checkbox("Line Noise", value=True, key='loaded_line')
                            loaded_include_emg = st.checkbox("EMG Noise", value=True, key='loaded_emg')
                            loaded_include_ecg = st.checkbox("ECG Artefact", value=False, key='loaded_ecg')
                            loaded_include_movement = st.checkbox("Movement Artefact", value=False, key='loaded_movement')
                            loaded_include_tms = st.checkbox("TMS Artefact", value=True, key='loaded_tms')
                        
                        st.info("💡 **Tip:** For loaded waveforms with existing baseline noise, use higher SNR (20-30 dB) and fewer noise types.")
                    
                    if st.button("📋 Use Loaded Waveform Directly as Template", type="primary", key='use_loaded'):
                        # Use loaded waveform data from session state
                        if hasattr(st.session_state, 'loaded_waveform') and hasattr(st.session_state, 'current_loaded_data'):
                            time_template, signal_template, fs_template = st.session_state.loaded_waveform
                            
                            # Normalize to desired amplitude
                            normalized_signal = signal_template / np.max(np.abs(signal_template)) * params['peak_amplitude']
                            
                            # Add noise with USER-CONFIGURED parameters
                            noise_gen_temp = NoiseGenerator(sampling_rate=int(fs_template))
                            EP_noisy_temp = noise_gen_temp.add_composite_noise(
                                normalized_signal,
                                time_template,
                                snr_db=loaded_snr_db,  # User-configured SNR
                                include_line=loaded_include_line,  # User-configured
                                include_emg=loaded_include_emg,    # User-configured
                                include_ecg=loaded_include_ecg,    # User-configured
                                include_movement=loaded_include_movement,  # User-configured
                                include_tms=loaded_include_tms     # User-configured
                            )
                            
                            # Apply polarity inversion to final signal
                            if invert_polarity:
                                EP_clean = -EP_clean
                                EP_noisy = -EP_noisy

                            # Store as generated signal with both clean and noisy versions
                            st.session_state.generated_signals = {
                                'time': time_template,
                                'EP_clean': normalized_signal,
                                'EP_noisy': EP_noisy_temp,
                                'sampling_rate': int(fs_template),
                                'parameters': {
                                    'amplitude': params['peak_amplitude'],
                                    'duration': params['duration_ms'],
                                    'onset_latency': params['onset_latency_ms'],
                                    'snr_db': loaded_snr_db,  # Store user SNR
                                    'EP_type': f"Template from {file_metadata.get('filename', 'loaded file')}",
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
    
    # ========== STANDARD EP PARAMETERS ==========
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EP Parameters")
        
        # Use standard defaults
        default_sampling_rate = 5000
        default_amplitude = 1.0
        default_onset = 20
        default_rise = 8
        default_decay = 15
        default_duration = 30
        default_EP_type = 'Standard'
        
        sampling_rate = st.number_input("Sampling Rate (Hz)", 
                                       min_value=500, max_value=20000, 
                                       value=int(default_sampling_rate), step=100,
                                       key='gen_sampling_rate')
        
        EP_amplitude = st.slider("EP Amplitude (mV)", 
                                 min_value=0.1, max_value=5.0, 
                                 value=float(default_amplitude), step=0.1,
                                 key='gen_amplitude')
        
        EP_duration = st.slider("EP Duration (ms)", 
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
        
        EP_type = st.selectbox("EP Type", 
                               ["Mono-Phasic", "Bi-phasic", "Tri-phasic"],
                               index=0,
                               key='gen_EP_type')
        
        # Polarity control
        # Polarity Control (Universal - applies to all EP types)
        invert_polarity = st.checkbox(
            "⬇️ Invert polarity",
            value=False,
            key='invert_polarity',
            help="Flip waveform upside down for inverted electrode montages"
        )
        
        # Always use physiologically realistic onset with pre-onset dip
        include_onset_dip = True
        
        # EP Variability Controls
        with st.expander("🔄 EP Variability", expanded=True):
            st.markdown("""
            Add realistic trial-to-trial variations:
            - **Amplitude**: ±20-30% (neuronal recruitment)
            - **Latency**: ±1-3 ms (conduction time)
            - **Duration**: ±10-15% (motor unit populations)
            """)
            
            enable_EP_variability = st.checkbox("✓ Enable EP Variability", value=False, key='enable_EP_var')
            
            if enable_EP_variability:
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    amplitude_variability = st.slider("Amplitude (±%)", 0, 50, 30, 5, key='amp_var') / 100.0
                with col_v2:
                    latency_variability = st.slider("Latency (±ms)", 0.0, 5.0, 2.0, 0.5, key='lat_var')
                with col_v3:
                    duration_variability = st.slider("Duration (±%)", 0, 30, 10, 5, key='dur_var') / 100.0
            else:
                amplitude_variability = 0.0
                latency_variability = 0.0
                duration_variability = 0.0
        
        # ========== ADVANCED PHASE CONTROLS ==========
        if EP_type == "Bi-phasic":
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
        
        if EP_type == "Tri-phasic":
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
        include_line = st.checkbox("Line Noise", value=True)
        include_emg = st.checkbox("EMG Noise", value=True)
        include_ecg = st.checkbox("ECG Artefact", value=False)
        include_movement = st.checkbox("Movement Artefact", value=False)
        include_tms = st.checkbox("TMS Artefact", value=True)
        
        if include_line:
            line_freq = st.selectbox("Line Frequency", [50, 60], index=0)
        else:
            line_freq = 50
            
    # Generate signal button
    if st.button("🎯 Generate Signal", type="primary"):
        with st.spinner("Generating signals..."):
            # Initialize generators
            EP_gen = EPGenerator(sampling_rate=sampling_rate)
            noise_gen = NoiseGenerator(sampling_rate=sampling_rate)
            
            # Generate clean EP
            if EP_type == "Mono-Phasic":
                time, EP_clean = EP_gen.generate_EP(
                    amplitude=EP_amplitude,
                    duration=EP_duration/1000,
                    onset_latency=onset_latency/1000,
                    rise_time=rise_time/1000,
                    decay_time=decay_time/1000,
                    asymmetry=asymmetry
                )
            elif EP_type == "Bi-phasic":
                if use_advanced_biphasic:
                    # Use advanced bi-phasic method with phase-specific controls
                    time, EP_clean = EP_gen.generate_biphasic_advanced(
                        phase1_amplitude=EP_amplitude,
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
                    time, EP_clean = EP_gen.generate_biphasic_EP(
                        amplitude=EP_amplitude,
                        onset_latency=onset_latency/1000,
                        phase1_duration=rise_time/1000 * 1.2,
                        phase2_duration=decay_time/1000 * 1.0,
                        phase2_amplitude_ratio=0.8,
                        include_onset_dip=include_onset_dip
                    )
            elif EP_type == "Tri-phasic":
                if use_advanced_triphasic:
                    # Use advanced tri-phasic method with phase-specific controls
                    time, EP_clean = EP_gen.generate_triphasic_advanced(
                        phase1_amplitude=EP_amplitude,
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
                    time, EP_clean = EP_gen.generate_triphasic_EP(
                        amplitude=EP_amplitude,
                        onset_latency=onset_latency/1000,
                        phase1_duration=rise_time/1000 * 1.0,
                        phase2_duration=decay_time/1000 * 1.2,
                        phase3_duration=rise_time/1000 * 1.3,
                        phase2_ratio=0.75,
                        phase3_ratio=0.4,
                        include_onset_dip=include_onset_dip
                    )
            elif EP_type == "Double Peak":
                time, EP_clean = EP_gen.generate_double_peak_EP(
                    amplitude1=EP_amplitude,
                    amplitude2=EP_amplitude * 0.6,
                    onset_latency=onset_latency/1000
                )
            else:  # With Baseline EMG
                time, EP_clean = EP_gen.generate_EP(
                    amplitude=EP_amplitude,
                    duration=EP_duration/1000,
                    onset_latency=onset_latency/1000,
                    rise_time=rise_time/1000,
                    decay_time=decay_time/1000,
                    asymmetry=asymmetry
                )
                EP_clean = EP_gen.add_baseline_emg(EP_clean, time, 
                                                    emg_amplitude=0.05,
                                                    onset_latency=onset_latency/1000)
            
            # Add noise
            EP_noisy = noise_gen.add_composite_noise(
                EP_clean, time,
                snr_db=snr_db,
                include_line=include_line,
                include_emg=include_emg,
                include_ecg=include_ecg,
                include_movement=include_movement,
                include_tms=include_tms
            )
            
            # Apply polarity inversion to final signals (excluding TMS artefact)
            # st.write(f"DEBUG BEFORE: invert_polarity={invert_polarity}, EP_clean[peak]={EP_clean[np.argmax(np.abs(EP_clean))]:.3f}")
            if invert_polarity:
                # Find TMS artefact location (at t=0)
                t0_idx = np.argmin(np.abs(time))
                
                # Extract TMS artefact if present (check if include_tms was True)
                if include_tms:
                    # Store artefact value
                    artefact_value = EP_noisy[t0_idx]
                    
                    # Invert the signals
                    EP_clean = -EP_clean
                    EP_noisy = -EP_noisy
                    
                    # Restore TMS artefact to original polarity (downward spike)
                    # The artefact was negative, got inverted to positive, so restore to negative
                    EP_noisy[t0_idx] = artefact_value
                else:
                    # No artefact, just invert
                    EP_clean = -EP_clean
                    EP_noisy = -EP_noisy
                
                # st.write(f"DEBUG AFTER INVERSION: EP_clean[peak]={EP_clean[np.argmax(np.abs(EP_clean))]:.3f}")
            
            # Store in session state
            st.session_state.generated_signals = {
                'time': time,
                'EP_clean': EP_clean,
                'EP_noisy': EP_noisy,
                'sampling_rate': sampling_rate,
                'parameters': {
                    'EP_type': EP_type,
                    'amplitude': EP_amplitude,
                    'duration': EP_duration,
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
                    'line_freq': line_freq,
                    'invert_polarity': invert_polarity,
                    'include_onset_dip': include_onset_dip,
                    'variability_enabled': enable_EP_variability,
                    'amplitude_variability': amplitude_variability,
                    'latency_variability': latency_variability,
                    'duration_variability': duration_variability
                }
            }
            
        st.success("✅ Signal generated successfully!")
        
        # Show variability and polarity status
        if enable_EP_variability:
            st.info(f"✓ EP VARIABILITY ENABLED: Amplitude ±{amplitude_variability*100:.0f}%, "
                   f"Latency ±{latency_variability:.1f}ms, Duration ±{duration_variability*100:.0f}%")
        else:
            st.warning("⚠️ EP Variability DISABLED - Enable above for realistic testing")
        
        if invert_polarity:
            st.info("⬇️ POLARITY INVERTED - Waveform is flipped upside down")
    
    # Display generated signals
    if st.session_state.generated_signals is not None:
        st.subheader("Generated Signals")
        
        data = st.session_state.generated_signals
        
        # st.write(f"DEBUG DISPLAY: Retrieved EP_clean[peak]={data['EP_clean'][np.argmax(np.abs(data['EP_clean']))]:.3f}")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Plot clean EP
        axes[0].plot(data['time']*1000, data['EP_clean'], 'b-', linewidth=1.5, label='Clean EP')
        axes[0].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1, label='TMS Stimulus')
        axes[0].set_ylabel('Amplitude (mV)')
        axes[0].set_title('Clean EP Signal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot noisy EP
        axes[1].plot(data['time']*1000, data['EP_noisy'], 'r-', linewidth=0.8, alpha=0.7, label='Noisy EP')
        axes[1].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1, label='TMS Stimulus')
        axes[1].set_ylabel('Amplitude (mV)')
        axes[1].set_title('Noisy EP Signal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot overlay
        axes[2].plot(data['time']*1000, data['EP_clean'], 'b-', linewidth=1.5, alpha=0.7, label='Clean')
        axes[2].plot(data['time']*1000, data['EP_noisy'], 'r-', linewidth=0.8, alpha=0.5, label='Noisy')
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
        
        # ========== FREQUENCY ANALYSIS SECTION ==========
        st.divider()
        st.subheader("📊 Frequency Content Analysis")
        
        with st.expander("ℹ️ Understanding Frequency Analysis", expanded=False):
            st.markdown("""
            **Power Spectral Density (PSD)** shows how signal power is distributed across frequencies.
            
            **What to look for:**
            - **EP energy**: Typically concentrated in 20-150 Hz range
            - **Line noise**: Sharp peak at 50/60 Hz (if enabled)
            - **EMG noise**: Broadband energy across 20-500 Hz
            - **Low-frequency drift**: Energy below 10 Hz (movement artifacts)
            - **High-frequency noise**: Energy above 300 Hz (amplifier noise, TMS artifact)
            
            **Interpretation:**
            - Clean EP should have most power in 20-150 Hz
            - Noisy signal will show elevated power across broader frequency range
            - The goal of filtering is to preserve EP frequencies while removing noise frequencies
            """)
        
        col_freq1, col_freq2 = st.columns([3, 1])
        
        with col_freq1:
            freq_range_max = st.slider(
                "Maximum frequency to display (Hz)",
                min_value=100,
                max_value=int(data['sampling_rate'] / 2),
                value=min(500, int(data['sampling_rate'] / 2)),
                step=50,
                key='freq_display_max',
                help="Adjust to focus on different frequency ranges"
            )
        
        with col_freq2:
            st.write("**Plot Options:**")
            show_db_scale = st.checkbox("Use dB scale", value=True, key='freq_db')
            show_ep_band = st.checkbox("Highlight EP band", value=True, key='freq_highlight')
        
        if st.button("🎨 Generate Frequency Analysis", type="primary", key='freq_analysis_btn'):
            with st.spinner("Computing frequency spectra..."):
                from scipy import signal as scipy_signal
                
                # Compute Power Spectral Density using Welch's method
                # Welch's method: More robust than raw FFT, reduces noise via averaging
                nperseg = min(256, len(data['EP_clean']) // 4)  # Window length
                
                # Clean signal PSD
                freqs_clean, psd_clean = scipy_signal.welch(
                    data['EP_clean'],
                    fs=data['sampling_rate'],
                    nperseg=nperseg,
                    scaling='density'
                )
                
                # Noisy signal PSD
                freqs_noisy, psd_noisy = scipy_signal.welch(
                    data['EP_noisy'],
                    fs=data['sampling_rate'],
                    nperseg=nperseg,
                    scaling='density'
                )
                
                # Create dual-panel figure
                fig, axes = plt.subplots(2, 1, figsize=(14, 10))
                
                # ===== PANEL 1: PSD Comparison =====
                ax1 = axes[0]
                
                # Plot PSDs
                if show_db_scale:
                    # Convert to dB (10*log10 for power)
                    psd_clean_db = 10 * np.log10(psd_clean + 1e-12)  # Avoid log(0)
                    psd_noisy_db = 10 * np.log10(psd_noisy + 1e-12)
                    
                    ax1.plot(freqs_clean, psd_clean_db, 'b-', linewidth=2, alpha=0.8, 
                            label='Clean EP', zorder=10)
                    ax1.plot(freqs_noisy, psd_noisy_db, 'r-', linewidth=2, alpha=0.7, 
                            label='Noisy EP', zorder=9)
                    
                    ylabel = 'Power Spectral Density (dB/Hz)'
                else:
                    ax1.plot(freqs_clean, psd_clean, 'b-', linewidth=2, alpha=0.8, 
                            label='Clean EP', zorder=10)
                    ax1.plot(freqs_noisy, psd_noisy, 'r-', linewidth=2, alpha=0.7, 
                            label='Noisy EP', zorder=9)
                    
                    ylabel = 'Power Spectral Density (V²/Hz)'
                    ax1.set_yscale('log')  # Log scale for power in linear units
                
                # Highlight EP frequency band (20-150 Hz)
                if show_ep_band:
                    ax1.axvspan(20, 150, color='green', alpha=0.15, 
                               label='Typical EP Band (20-150 Hz)', zorder=1)
                
                # Mark line noise if present
                if data['parameters'].get('noise_types', {}).get('line', False):
                    line_freq = data['parameters'].get('line_freq', 50)
                    ax1.axvline(line_freq, color='orange', linestyle='--', linewidth=2, 
                               alpha=0.7, label=f'Line Noise ({line_freq} Hz)', zorder=8)
                    # Mark harmonics
                    for harmonic in [2, 3]:
                        ax1.axvline(line_freq * harmonic, color='orange', linestyle=':', 
                                   linewidth=1.5, alpha=0.5, zorder=8)
                
                ax1.set_xlim([0, freq_range_max])
                ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
                ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                ax1.set_title('A) Power Spectral Density - Clean vs Noisy Signal', 
                             fontsize=14, fontweight='bold')
                ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
                ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                
                # ===== PANEL 2: Noise Power Distribution =====
                ax2 = axes[1]
                
                # Calculate noise power (difference between noisy and clean)
                noise_psd = psd_noisy - psd_clean
                noise_psd[noise_psd < 0] = 0  # Ensure non-negative
                
                # Calculate actual SNR to detect negligible noise
                # If SNR is very high and no noise sources enabled, noise is just numerical artifacts
                ep_band_mask = (freqs_clean >= 20) & (freqs_clean <= 150)
                signal_power_ep = np.mean(psd_clean[ep_band_mask])
                noise_power_ep = np.mean(noise_psd[ep_band_mask])
                
                # Calculate actual SNR in dB
                if noise_power_ep > 1e-20:
                    actual_snr_db = 10 * np.log10(signal_power_ep / noise_power_ep)
                else:
                    actual_snr_db = 100  # Essentially infinite SNR
                
                # Check if any noise sources are enabled
                noise_types = data['parameters'].get('noise_types', {})
                any_noise_enabled = any(noise_types.values())
                
                # Determine if noise is negligible (SNR > 35 dB and no noise sources)
                noise_is_negligible = (actual_snr_db > 35) and not any_noise_enabled
                
                if noise_is_negligible:
                    # Display message instead of misleading plot
                    ax2.text(0.5, 0.5, 
                            '✓ Noise is negligible\n\n'
                            f'Actual SNR: {actual_snr_db:.1f} dB\n'
                            'All noise sources: OFF\n\n'
                            'The small differences between Clean and Noisy signals\n'
                            'are due to numerical precision, not real noise.',
                            transform=ax2.transAxes,
                            fontsize=14,
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
                    ax2.set_xlim([0, freq_range_max])
                    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Noise Power (dB/Hz)' if show_db_scale else 'Noise Power (V²/Hz)', 
                                  fontsize=12, fontweight='bold')
                    ax2.set_title('B) Added Noise - Frequency Distribution', 
                                 fontsize=14, fontweight='bold')
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                else:
                    # Plot actual noise distribution
                    if show_db_scale:
                        noise_psd_plot = 10 * np.log10(noise_psd + 1e-12)
                    else:
                        noise_psd_plot = noise_psd
                    
                    ax2.fill_between(freqs_noisy, noise_psd_plot, color='red', alpha=0.4, 
                                    label='Added Noise Power')
                    ax2.plot(freqs_noisy, noise_psd_plot, 'r-', linewidth=2, alpha=0.8)
                    
                    # Highlight different noise regions
                    ax2.axvspan(0, 10, color='purple', alpha=0.1, label='Low-freq drift (<10 Hz)')
                    ax2.axvspan(20, 150, color='green', alpha=0.1, label='EP band (20-150 Hz)')
                    ax2.axvspan(300, freq_range_max, color='blue', alpha=0.1, 
                               label='High-freq noise (>300 Hz)')
                    
                    ax2.set_xlim([0, freq_range_max])
                    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Noise Power (dB/Hz)' if show_db_scale else 'Noise Power (V²/Hz)', 
                                  fontsize=12, fontweight='bold')
                    ax2.set_title('B) Added Noise - Frequency Distribution', 
                                 fontsize=14, fontweight='bold')
                    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
                    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                    
                    if not show_db_scale:
                        ax2.set_yscale('log')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add download button
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="💾 Download Frequency Analysis (300 DPI)",
                    data=buf,
                    file_name="frequency_analysis_300dpi.png",
                    mime="image/png",
                    key='download_freq_analysis'
                )
                
                plt.close(fig)
                
                # ===== QUANTITATIVE METRICS =====
                st.subheader("📈 Frequency Metrics")
                
                # Calculate metrics for different frequency bands
                def calculate_band_power(freqs, psd, f_low, f_high):
                    """Calculate power in a frequency band."""
                    mask = (freqs >= f_low) & (freqs <= f_high)
                    return np.trapz(psd[mask], freqs[mask])
                
                # Define frequency bands
                bands = {
                    'Low-frequency (<10 Hz)': (0, 10),
                    'EP band (20-150 Hz)': (20, 150),
                    'EMG range (20-500 Hz)': (20, 500),
                    'High-frequency (>300 Hz)': (300, min(1000, data['sampling_rate']/2))
                }
                
                metrics_data = []
                
                for band_name, (f_low, f_high) in bands.items():
                    power_clean = calculate_band_power(freqs_clean, psd_clean, f_low, f_high)
                    power_noisy = calculate_band_power(freqs_noisy, psd_noisy, f_low, f_high)
                    power_increase = ((power_noisy - power_clean) / power_clean * 100) if power_clean > 0 else 0
                    
                    metrics_data.append({
                        'Frequency Band': band_name,
                        'Clean Power': f'{power_clean:.6f}',
                        'Noisy Power': f'{power_noisy:.6f}',
                        'Power Increase (%)': f'{power_increase:.1f}%'
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Add note if noise is negligible
                if noise_is_negligible:
                    st.info("""
                    ℹ️ **Note on Power Increase Values**: With very high SNR (>35 dB) and all noise sources 
                    disabled, the small "power increase" values shown above are due to numerical precision 
                    in the spectral estimation, not real added noise. The noisy signal is effectively 
                    identical to the clean signal.
                    """)
                
                # Dominant frequency analysis
                col_dom1, col_dom2 = st.columns(2)
                
                with col_dom1:
                    st.write("**Clean EP - Dominant Frequency:**")
                    # Focus on EP band for clean signal
                    ep_mask_clean = (freqs_clean >= 20) & (freqs_clean <= 150)
                    dominant_freq_clean = freqs_clean[ep_mask_clean][np.argmax(psd_clean[ep_mask_clean])]
                    st.metric("Peak frequency", f"{dominant_freq_clean:.1f} Hz")
                    
                    # Calculate bandwidth (frequencies containing 90% of power in EP band)
                    ep_power_clean = psd_clean[ep_mask_clean]
                    cumsum_power = np.cumsum(ep_power_clean) / np.sum(ep_power_clean)
                    f_low_idx = np.argmax(cumsum_power >= 0.05)  # 5th percentile
                    f_high_idx = np.argmax(cumsum_power >= 0.95)  # 95th percentile
                    bandwidth = freqs_clean[ep_mask_clean][f_high_idx] - freqs_clean[ep_mask_clean][f_low_idx]
                    st.metric("90% Power Bandwidth", f"{bandwidth:.1f} Hz")
                
                with col_dom2:
                    st.write("**Noisy EP - Dominant Frequency:**")
                    ep_mask_noisy = (freqs_noisy >= 20) & (freqs_noisy <= 150)
                    dominant_freq_noisy = freqs_noisy[ep_mask_noisy][np.argmax(psd_noisy[ep_mask_noisy])]
                    st.metric("Peak frequency", f"{dominant_freq_noisy:.1f} Hz")
                    
                    # Spectral centroid (weighted average frequency)
                    spectral_centroid = np.sum(freqs_noisy[ep_mask_noisy] * psd_noisy[ep_mask_noisy]) / np.sum(psd_noisy[ep_mask_noisy])
                    st.metric("Spectral Centroid", f"{spectral_centroid:.1f} Hz")
                
                st.success("✅ Frequency analysis complete!")
                
                # Interpretation guide
                with st.expander("📚 How to Interpret These Results"):
                    st.markdown(f"""
                    ### Your Signal Analysis:
                    
                    **Clean EP Characteristics:**
                    - Dominant frequency: {dominant_freq_clean:.1f} Hz
                    - This is {"within" if 30 <= dominant_freq_clean <= 100 else "outside"} the typical range (30-100 Hz for limb muscles)
                    - 90% of clean EP power is contained within {bandwidth:.1f} Hz bandwidth
                    
                    **Noise Impact:**
                    - SNR: {data['parameters']['snr_db']:.0f} dB
                    - {"High SNR (>20 dB): Noise minimally affects frequency content" if data['parameters']['snr_db'] > 20 else ""}
                    - {"Moderate SNR (10-20 dB): Noticeable broadband noise" if 10 <= data['parameters']['snr_db'] <= 20 else ""}
                    - {"Low SNR (<10 dB): Substantial noise across all frequencies" if data['parameters']['snr_db'] < 10 else ""}
                    
                    ### Filter Recommendations:
                    
                    **Based on your signal:**
                    - **High-pass cutoff**: Should be below {dominant_freq_clean:.0f} Hz (e.g., 10-20 Hz)
                    - **Low-pass cutoff**: Should be above {dominant_freq_clean + bandwidth/2:.0f} Hz (e.g., 200-500 Hz)
                    - Preserving {dominant_freq_clean - bandwidth/2:.0f}-{dominant_freq_clean + bandwidth/2:.0f} Hz range is critical
                    
                    **Noise removal targets:**
                    """ + (f"- Line noise at {data['parameters'].get('line_freq', 50)} Hz: Use notch filter\n" if data['parameters'].get('noise_types', {}).get('line') else "") + """
                    - Low-frequency drift (<10 Hz): High-pass filter
                    - High-frequency noise (>300 Hz): Low-pass filter
                    - EMG contamination (20-500 Hz): Challenging - overlaps with EP frequencies
                    
                    ### Quality Checks:
                    
                    ✅ **Good signal** if:
                    - Most clean EP power is in 20-150 Hz (typical for EPs)
                    - Dominant frequency is physiologically realistic (30-100 Hz)
                    - Narrow bandwidth (<100 Hz) indicates clear, well-defined response
                    
                    ⚠️ **Check if**:
                    - Dominant frequency is very low (<20 Hz) or very high (>150 Hz)
                    - Bandwidth is very wide (>200 Hz) - may indicate multiple components
                    - High power at unexpected frequencies (could indicate artifacts)
                    """)
                
                st.info("""
                💡 **Pro Tip**: Compare this frequency analysis with your filtered signals in the 
                Filter Testing tab to verify that your chosen filters preserve the EP frequency 
                content while removing noise frequencies. The time-frequency analysis in Tab 2 
                provides even more detailed spectral information.
                """)

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
        filters = EPFilters(sampling_rate=data['sampling_rate'])
        
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
            
            use_notch = st.checkbox("Notch Filter", value=False)
            
            if st.button("🔧 Apply Filter", type="primary"):
                # Prepare filter parameters
                filter_params = {
                    'filter_type': filter_type,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'order': order,
                    'notch_enabled': use_notch,
                    'notch_freq': data['parameters'].get('line_freq', 50)
                }
                
                if filter_type == 'moving_average':
                    filter_params['window_size'] = window_size
                elif filter_type == 'savitzky_golay':
                    filter_params['window_length'] = window_length
                    filter_params['polyorder'] = polyorder
                
                # Apply filter
                with st.spinner("Applying filter..."):
                    EP_filtered = filters.apply_filter_cascade(
                        data['EP_noisy'], 
                        filter_params
                    )
                    
                    # Calculate metrics
                    metrics_calc = EPMetrics(sampling_rate=data['sampling_rate'])
                    metrics = metrics_calc.calculate_all_metrics(
                        data['EP_clean'],
                        EP_filtered,
                        data['time']
                    )
                    
                    # Store results
                    result = {
                        'filter_params': filter_params,
                        'filtered_signal': EP_filtered,
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
                if len(result['filtered_signal']) != len(data['EP_clean']):
                    # Trim to match shorter length
                    min_len = min(len(result['filtered_signal']), len(data['EP_clean']))
                    filtered_plot = result['filtered_signal'][:min_len]
                    clean_plot = data['EP_clean'][:min_len]
                    noisy_plot = data['EP_noisy'][:min_len]
                    time_plot = data['time'][:min_len]
                else:
                    filtered_plot = result['filtered_signal']
                    clean_plot = data['EP_clean']
                    noisy_plot = data['EP_noisy']
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
                           f"{m['rmse_EP']:.3f} mV")
                
                # Detailed metrics table
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Amplitude Error (%)',
                        'Peak Latency Error (ms)',
                        'Onset Error (ms)',
                        'AUC Error (%)',
                        'Correlation',
                        'RMSE (EP window)',
                        'Baseline Std (mV)',
                        'SNR (dB)'
                    ],
                    'Value': [
                        f"{m['amplitude_error_pct']:.2f}",
                        f"{m['peak_latency_error_ms']:.3f}",
                        f"{m['onset_error_ms']:.3f}",
                        f"{m['auc_error_pct']:.2f}",
                        f"{m['correlation']:.4f}",
                        f"{m['rmse_EP']:.4f}",
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
                            data['EP_clean'],
                            data['EP_noisy'],
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
                        - **Ground truth (A):** Shows natural EP frequency content
                        - **Noisy signal (B):** Shows added noise across all frequencies
                        - **Filtered signal (C):** Should match (A) in EP window, reduced elsewhere
                        - **Good filter:** Preserves EP spectral structure, removes noise frequencies
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
                default=["butterworth"]
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
                default=[1, 2, 4],
                help="Select designed orders. Effective orders = 2× these values with zero-phase filtering."
            )
            
            st.write("**Highpass Cutoffs (Hz)**")
            highpass_cutoffs = st.text_input("Comma-separated values", "10, 20")
            
            st.write("**Lowpass Cutoffs (Hz)**")
            lowpass_cutoffs = st.text_input("Comma-separated values (lowpass)", "500, 1000")
            
            st.info(f"ℹ️ **Note:** With sampling rate of {st.session_state.generated_signals['sampling_rate']} Hz, "
                   f"cutoff frequencies must be below {st.session_state.generated_signals['sampling_rate']/2:.0f} Hz (Nyquist limit).")
            
            include_notch_batch = st.checkbox("Test with/without notch filter", value=False)
            
        with col2:
            st.subheader("Analysis Options")
            
            n_iterations = st.slider("Iterations per configuration", min_value=1, max_value=max_batch_iterations, value=default_batch_iterations, step=1)
            
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
            
            filters = EPFilters(sampling_rate=data['sampling_rate'])
            metrics_calc = EPMetrics(sampling_rate=data['sampling_rate'])
            EP_gen = EPGenerator(sampling_rate=data['sampling_rate'])
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
                                        'notch_freq': data['parameters'].get('line_freq', 50)
                                    })
                        else:
                            for notch in ([True, False] if include_notch_batch else [False]):
                                configurations.append({
                                    'filter_type': filter_type,
                                    'lowcut': hp,
                                    'highcut': lp,
                                    'order': 4,
                                    'notch_enabled': notch,
                                    'notch_freq': data['parameters'].get('line_freq', 50)
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
                        # Generate EP with variability if enabled
                        if data['parameters'].get('variability_enabled', False):
                            import numpy as np
                            from signal_generator import EPGenerator
                            
                            # Set seed FIRST for reproducibility
                            np.random.seed(current_test)
                            
                            temp_gen = EPGenerator(sampling_rate=data['sampling_rate'])
                            amp_temp, lat_temp, dur_temp = temp_gen.apply_EP_variability(
                                base_amplitude=data['parameters']['amplitude'],
                                base_latency=data['parameters']['onset_latency'] / 1000,
                                base_duration=data['parameters']['duration'] / 1000,
                                amplitude_variability=data['parameters']['amplitude_variability'],
                                latency_variability=data['parameters']['latency_variability'] / 1000,
                                duration_variability=data['parameters']['duration_variability']
                            )
                            
                            # Store actual randomized values for exact reproducibility
                            actual_amplitude = amp_temp
                            actual_latency = lat_temp
                            actual_duration = dur_temp
                            
                            time_iter, EP_clean_iter = temp_gen.generate_EP(
                                amplitude=amp_temp,
                                duration=dur_temp,
                                onset_latency=lat_temp,
                                rise_time=data['parameters'].get('rise_time', 8) / 1000,
                                decay_time=data['parameters'].get('decay_time', 15) / 1000
                            )
                        else:
                            time_iter = data['time']
                            EP_clean_iter = data['EP_clean']
                        
                        # Generate noisy signal
                        # CRITICAL FIX: Use actual noise settings from GUI, not hardcoded values
                        noise_types = data['parameters'].get('noise_types', {})
                        EP_noisy = noise_gen.add_composite_noise(
                            EP_clean_iter, time_iter,
                            snr_db=data['parameters']['snr_db'],
                            include_line=noise_types.get('line', True),
                            include_emg=noise_types.get('emg', True),
                            include_ecg=noise_types.get('ecg', False),
                            include_movement=noise_types.get('movement', False),
                            include_tms=noise_types.get('tms', False)
                        )
                        
                        # Apply filter
                        EP_filtered = filters.apply_filter_cascade(EP_noisy, config)
                        
                        # Calculate metrics
                        metrics = metrics_calc.calculate_all_metrics(
                            EP_clean_iter,
                            EP_filtered,
                            time_iter
                        )
                        
                        # Store results
                        # Store result with exact iteration parameters
                        iteration_params = {}
                        if data['parameters'].get('variability_enabled', False):
                            iteration_params = {
                                'actual_amplitude': actual_amplitude,
                                'actual_latency': actual_latency * 1000,  # Convert to ms
                                'actual_duration': actual_duration * 1000,  # Convert to ms
                            }
                        
                        result = {**config, **metrics, 'iteration': iteration, 'seed_used': current_test, **iteration_params}
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
                
                # Add log-transformed metrics for statistical analysis
                metrics_calc = EPMetrics()  # Create instance for transform methods
                metrics_to_transform = ['amplitude_error_pct', 'peak_latency_error_ms', 'onset_error_ms', 'rmse_EP']
                
                for metric in metrics_to_transform:
                    if metric in results_df.columns:
                        results_df[f'{metric}_log'] = results_df[metric].apply(
                            lambda x: metrics_calc.apply_log_transform(np.array([x]))[0]
                        )
                
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
                'onset_error_ms': ['mean', 'std'],
                'correlation_z': ['mean', 'std'],  # Fisher's z-transformed correlation
                'rmse_EP': ['mean', 'std']
            }).round(3)
            
            # Rename columns for clarity
            summary.columns = [' '.join(col).strip() for col in summary.columns.values]
            summary = summary.rename(columns={
                'correlation_z mean': 'correlation_z mean (Fisher z)',
                'correlation_z std': 'correlation_z std (Fisher z)'
            })
            
            st.subheader("Summary Statistics")
            st.dataframe(summary, width='stretch')
            
            st.info("""
            **Note**: Correlation values are reported as Fisher's z-transformed values for valid statistical analysis.
            To convert back to r: r = tanh(z)
            """)
            
            # Test for bimodality in distributions
            if 'config_label' in df.columns:
                st.subheader("📊 Distribution Analysis: Bimodality Testing")
                
                metrics_calc = EPMetrics()
                bimodal_results = []
                
                for config in df['config_label'].unique():
                    config_data = df[df['config_label'] == config]['amplitude_error_pct'].values
                    bc_result = metrics_calc.test_bimodality_coefficient(config_data)
                    
                    bimodal_results.append({
                        'Configuration': config,
                        'Bimodality Coef': round(bc_result['bc'], 3) if not np.isnan(bc_result['bc']) else 'N/A',
                        'Bimodal?': '⚠️ Yes' if bc_result['is_bimodal'] else '✓ No',
                        'Skewness': round(bc_result['skewness'], 3) if not np.isnan(bc_result['skewness']) else 'N/A',
                        'Kurtosis': round(bc_result['kurtosis'], 3) if not np.isnan(bc_result['kurtosis']) else 'N/A'
                    })
                
                bimodal_df = pd.DataFrame(bimodal_results)
                st.dataframe(bimodal_df, width='stretch')
                
                with st.expander("ℹ️ Understanding Bimodality Testing"):
                    st.markdown("""
                    **Bimodality Coefficient (BC)**: BC > 0.555 suggests a bimodal (two-peaked) distribution.
                    
                    **Formula**: BC = (skewness² + 1) / kurtosis
                    
                    **Interpretation**:
                    - **BC ≤ 0.555** (✓ No): Unimodal distribution - filter performs consistently
                    - **BC > 0.555** (⚠️ Yes): Bimodal distribution - filter may have distinct "good" and "bad" performance modes
                    
                    **Why it matters**:
                    - Bimodal distributions indicate inconsistent filter performance
                    - May suggest filter works differently for certain signal characteristics
                    - Could indicate presence of outliers or subgroups in your data
                    - May require stratified analysis or filter parameter adjustment
                    
                    **Skewness**: Measures asymmetry (0 = symmetric, + = right tail, - = left tail)
                    
                    **Kurtosis**: Measures tail heaviness (3 = normal, >3 = heavy tails, <3 = light tails)
                    """)
            
            # Statistical Assumptions Testing
            st.subheader("📈 Statistical Assumptions: Do You Need Log Transformation?")
            
            st.markdown("""
            **Purpose**: Determine if your data meets the assumptions for parametric tests (t-tests, ANOVA).
            - **Normality**: Data should be approximately normally distributed
            - **Homogeneity of variance**: Groups should have similar variances
            
            If assumptions are violated, log transformation may help!
            """)
            
            metrics_calc_stats = EPMetrics()
            metrics_to_test = ['amplitude_error_pct', 'peak_latency_error_ms', 'onset_error_ms', 'rmse_EP']
            
            normality_results = []
            
            for metric in metrics_to_test:
                if metric in df.columns:
                    # Test overall normality
                    metric_data = df[metric].dropna().values
                    
                    if len(metric_data) > 0:
                        norm_result = metrics_calc_stats.test_normality(metric_data)
                        
                        # Test log-transformed version if available
                        log_metric = f'{metric}_log'
                        if log_metric in df.columns:
                            log_data = df[log_metric].dropna().values
                            log_norm_result = metrics_calc_stats.test_normality(log_data)
                        else:
                            log_norm_result = None
                        
                        # Test homogeneity across configurations
                        groups = [group[metric].dropna().values for _, group in df.groupby('config_label')]
                        groups = [g for g in groups if len(g) >= 2]
                        
                        if len(groups) >= 2:
                            homog_result = metrics_calc_stats.test_homogeneity_of_variance(groups)
                        else:
                            homog_result = None
                        
                        normality_results.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Shapiro p-value': f"{norm_result['shapiro_p']:.4f}" if not np.isnan(norm_result['shapiro_p']) else 'N/A',
                            'Normal?': '✓ Yes' if norm_result['is_normal'] else '✗ No',
                            'Skewness': f"{norm_result['skewness']:.3f}",
                            'Recommendation': norm_result['recommendation'],
                            'Log Better?': '✓ Yes' if log_norm_result and log_norm_result['is_normal'] and not norm_result['is_normal'] else ('Already OK' if norm_result['is_normal'] else 'Maybe')
                        })
            
            if normality_results:
                assumptions_df = pd.DataFrame(normality_results)
                st.dataframe(assumptions_df, width='stretch')
                
                with st.expander("ℹ️ How to Interpret These Results"):
                    st.markdown("""
                    **Shapiro-Wilk p-value**:
                    - **p > 0.05** → Data IS normally distributed ✓
                    - **p < 0.05** → Data is NOT normal ✗
                    
                    **Skewness**:
                    - **-0.5 to +0.5**: Fairly symmetric (good!)
                    - **±0.5 to ±1**: Moderately skewed (consider transformation)
                    - **> ±1**: Highly skewed (transformation recommended)
                    
                    **Recommendation**:
                    - **"✓ Data is normally distributed"**: Use raw data for statistics
                    - **"⚠️ Highly skewed"**: Use log-transformed data instead
                    - **"⚠️ Non-normal but not skewed"**: Check for outliers or bimodality first
                    
                    **Log Better?**:
                    - **"✓ Yes"**: Log transformation fixes the normality issue - USE IT!
                    - **"Already OK"**: Raw data is fine - no transformation needed
                    - **"Maybe"**: Log doesn't fully fix it - consider non-parametric tests
                    
                    ### When to Use Log Transformation:
                    
                    **USE log-transformed data when**:
                    1. Shapiro p < 0.05 (not normal)
                    2. Skewness > 1 or < -1 (highly skewed)
                    3. "Log Better?" shows "✓ Yes"
                    
                    **USE raw data when**:
                    1. Shapiro p > 0.05 (normal)
                    2. Skewness between -0.5 and +0.5 (symmetric)
                    3. "Already OK" recommendation
                    
                    ### For Your Analysis:
                    
                    Look at the table above and for each metric:
                    - If "Normal? = ✓ Yes" → Use the regular metric in your stats
                    - If "Normal? = ✗ No" AND "Log Better? = ✓ Yes" → Use the `_log` version in your stats
                    - If both show "✗ No" → Consider non-parametric tests (Mann-Whitney, Kruskal-Wallis)
                    """)
            
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
                        ['amplitude_error_pct', 'amplitude_error_pct_log',
                         'peak_latency_error_ms', 'peak_latency_error_ms_log',
                         'onset_error_ms', 'onset_error_ms_log',
                         'correlation', 'rmse_EP', 'rmse_EP_log', 'baseline_std'],
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
                    elif metric_choice_box == 'amplitude_error_pct_log':
                        ref_value = st.slider("Reference value (log)", -3.0, 3.0, 0.0, 0.1, key='box_ref_val')
                    elif metric_choice_box == 'peak_latency_error_ms':
                        ref_value = st.slider("Reference value (ms)", -5.0, 5.0, 0.0, 0.1, key='box_ref_val')
                    elif metric_choice_box == 'peak_latency_error_ms_log':
                        ref_value = st.slider("Reference value (log)", -3.0, 3.0, 0.0, 0.1, key='box_ref_val')
                    elif metric_choice_box == 'onset_error_ms':
                        ref_value = st.slider("Reference value (ms)", -10.0, 10.0, 0.0, 0.5, key='box_ref_val')
                    elif metric_choice_box == 'onset_error_ms_log':
                        ref_value = st.slider("Reference value (log)", -3.0, 3.0, 0.0, 0.1, key='box_ref_val')
                    elif metric_choice_box == 'correlation':
                        ref_value = st.slider("Reference value", 0.0, 1.0, 0.95, 0.01, key='box_ref_val')
                    elif metric_choice_box in ['rmse_EP_log']:
                        ref_value = st.slider("Reference value (log)", -5.0, 0.0, -2.0, 0.1, key='box_ref_val')
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
                
                # REVIEWER FIX: Use z-scores for correlation analysis
                plot_metric = 'correlation_z' if metric_choice_box == 'correlation' else metric_choice_box
                
                # Check if metric exists in dataframe
                if plot_metric not in df.columns:
                    st.error(f"❌ Metric '{plot_metric}' not found in results. Available metrics: {', '.join(df.columns)}")
                    st.stop()
                
                for i, (config, group) in enumerate(df.groupby('config_label')):
                    positions.append(i)
                    # Filter out NaN values for this metric
                    metric_values = group[plot_metric].dropna().values
                    if len(metric_values) == 0:
                        st.warning(f"⚠️ No valid data for {config} in {plot_metric}")
                        metric_values = np.array([0])  # Placeholder to avoid empty box
                    box_data.append(metric_values)
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
                # REVIEWER FIX: Better labels for all metric types
                if metric_choice_box == 'correlation':
                    ylabel = "Fisher's z-score"
                elif '_log' in metric_choice_box:
                    base_metric = metric_choice_box.replace('_log', '').replace('_', ' ').title()
                    ylabel = f"{base_metric} (Log Transformed)"
                else:
                    ylabel = metric_choice_box.replace('_', ' ').title()
                
                ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                
                if metric_choice_box == 'correlation':
                    title_metric = "Fisher's z-score"
                elif '_log' in metric_choice_box:
                    title_metric = ylabel
                else:
                    title_metric = metric_choice_box.replace("_", " ").title()
                
                ax.set_title(f'{title_metric} Distribution Across Iterations', 
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
                
                col_overlay1, col_overlay2, col_overlay3 = st.columns(3)
                
                with col_overlay1:
                    st.write("**Filtered Signals:**")
                    show_individual = st.checkbox("Show individual iterations", value=False, key='overlay_indiv')
                    if show_individual:
                        alpha_iterations = st.slider("Iteration transparency", 0.05, 0.5, 0.15, 0.05, key='overlay_alpha')
                    else:
                        alpha_iterations = 0.2
                
                with col_overlay2:
                    st.write("**Ground Truth:**")
                    var_enabled = st.session_state.generated_signals['parameters'].get('variability_enabled', False) if st.session_state.generated_signals else False
                    
                    show_gt_variability = st.checkbox("Show GT variability", value=var_enabled, key='show_gt_var',
                                                     help="Show ±SD for variable ground truth EPs")
                    if show_gt_variability and var_enabled:
                        show_gt_sd = st.checkbox("Show GT ±SD envelope", value=True, key='show_gt_sd')
                        show_gt_individual = st.checkbox("Show individual GTs", value=False, key='show_gt_indiv')
                    else:
                        show_gt_sd = False
                        show_gt_individual = False
                
                with col_overlay3:
                    st.write("**Plot Options:**")
                    show_mean_only = st.checkbox("Mean lines only", value=False, key='overlay_mean')
                    show_sd_envelope = st.checkbox("Show ±SD envelope", value=True, key='overlay_sd')
                    overlay_dpi = st.selectbox("Export DPI", [150, 300, 600], index=1, key='overlay_dpi')
                
                if len(selected_configs) > 0 and st.button("🎨 Generate Multi-Filter Overlay", type="primary", key='multi_overlay_btn'):
                    with st.spinner(f"Generating overlay for {len(selected_configs)} configurations..."):
                        # Initialize generators
                        noise_gen = NoiseGenerator(sampling_rate=data['sampling_rate'])
                        filters_obj = EPFilters(sampling_rate=data['sampling_rate'])
                        
                        fig, ax = plt.subplots(figsize=(15, 9))
                        
                        # Collect GTs with matching seeds
                        temp_all_gt = []
                        if data["parameters"].get("variability_enabled", False):
                            from signal_generator import EPGenerator
                            import numpy as np
                            first_cfg = df[df["config_label"] == selected_configs[0]]
                            for _, row in first_cfg.iterrows():
                                tg = EPGenerator(data["sampling_rate"])
                                np.random.seed(row.get("seed_used", 0))
                                a,l,d = tg.apply_EP_variability(data["parameters"]["amplitude"],
                                    data["parameters"]["onset_latency"]/1000, data["parameters"]["duration"]/1000,
                                    data["parameters"]["amplitude_variability"],
                                    data["parameters"]["latency_variability"]/1000,
                                    data["parameters"]["duration_variability"])
                                
                                # Use correct EP type
                                EP_type = data["parameters"].get("EP_type", "Standard")
                                if EP_type == "Bi-phasic":
                                    _, mgt = tg.generate_biphasic_EP(a * (-1 if data['parameters'].get('invert_polarity', False) else 1), l, 
                                        data["parameters"].get("rise_time", 8)/1000*1.2,
                                        data["parameters"].get("decay_time", 15)/1000, 0.8,
                                        include_onset_dip=data['parameters'].get('include_onset_dip', True))
                                elif EP_type == "Tri-phasic":
                                    _, mgt = tg.generate_triphasic_EP(a * (-1 if data['parameters'].get('invert_polarity', False) else 1), l,
                                        data["parameters"].get("rise_time", 8)/1000,
                                        data["parameters"].get("decay_time", 15)/1000*1.2,
                                        data["parameters"].get("rise_time", 8)/1000*1.3, 0.75, 0.4,
                                        include_onset_dip=data['parameters'].get('include_onset_dip', True))
                                else:
                                    _, mgt = tg.generate_EP(a,d,l,
                                        data["parameters"].get("rise_time", 8)/1000,
                                        data["parameters"].get("decay_time", 15)/1000)
                                temp_all_gt.append(mgt)
                            mgt_mean = np.mean(temp_all_gt, axis=0)
                        else:
                            mgt_mean = data["EP_clean"]
                        if show_gt_variability and len(temp_all_gt) > 0:
                            if show_gt_individual:
                                for gt in temp_all_gt:
                                    ax.plot(data["time"]*1000, gt, "k-", lw=0.3, alpha=0.15, zorder=9500)
                            if show_gt_sd and len(temp_all_gt) > 1:
                                std_gt = np.std(temp_all_gt, axis=0)
                                ax.fill_between(data["time"]*1000, mgt_mean-std_gt, mgt_mean+std_gt,
                                              color="gray", alpha=0.15, zorder=9800)
                        # Plot ground truth with highest z-order
                        ax.plot(data["time"]*1000, mgt_mean, "k-", lw=3.5, alpha=1, label="Ground Truth (n="+str(len(temp_all_gt) if len(temp_all_gt)>0 else 1)+") ±SD", zorder=10000)
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
                                'notch_freq': data['parameters'].get('line_freq', 50)
                            }
                            
                            # Collect all filtered signals for this config
                            all_filtered = []
                            
                            for idx, row in config_data.iterrows():
                                # Regenerate EP with same seed as batch (for variability)
                                if data['parameters'].get('variability_enabled', False):
                                    import numpy as np
                                    from signal_generator import EPGenerator
                                    temp_gen_filter = EPGenerator(sampling_rate=data['sampling_rate'])
                                    np.random.seed(row.get('seed_used', row['iteration']))
                                    
                                    amp_f, lat_f, dur_f = temp_gen_filter.apply_EP_variability(
                                        data['parameters']['amplitude'],
                                        data['parameters']['onset_latency']/1000,
                                        data['parameters']['duration']/1000,
                                        data['parameters']['amplitude_variability'],
                                        data['parameters']['latency_variability']/1000,
                                        data['parameters']['duration_variability']
                                    )
                                    
                                    # Use correct EP type
                                    EP_type_f = data['parameters'].get('EP_type', 'Standard')
                                    if EP_type_f == 'Bi-phasic':
                                        time_f, EP_clean_f = temp_gen_filter.generate_biphasic_EP(
                                            amp_f * (-1 if data['parameters'].get('invert_polarity', False) else 1), lat_f,
                                            data['parameters'].get('rise_time', 8)/1000*1.2,
                                            data['parameters'].get('decay_time', 15)/1000, 0.8,
                                            include_onset_dip=data['parameters'].get('include_onset_dip', True))
                                    elif EP_type_f == 'Tri-phasic':
                                        time_f, EP_clean_f = temp_gen_filter.generate_triphasic_EP(
                                            amp_f * (-1 if data['parameters'].get('invert_polarity', False) else 1), lat_f,
                                            data['parameters'].get('rise_time', 8)/1000,
                                            data['parameters'].get('decay_time', 15)/1000*1.2,
                                            data['parameters'].get('rise_time', 8)/1000*1.3, 0.75, 0.4,
                                            include_onset_dip=data['parameters'].get('include_onset_dip', True))
                                    else:
                                        time_f, EP_clean_f = temp_gen_filter.generate_EP(
                                            amp_f, dur_f, lat_f,
                                            data['parameters'].get('rise_time', 8)/1000,
                                            data['parameters'].get('decay_time', 15)/1000)
                                else:
                                    time_f = data['time']
                                    EP_clean_f = data['EP_clean']
                                
                                # Generate noisy signal from correct EP
                                # CRITICAL FIX: Use actual noise settings from GUI, not hardcoded values
                                noise_types = data['parameters'].get('noise_types', {})
                                EP_noisy = noise_gen.add_composite_noise(
                                    EP_clean_f, time_f,
                                    snr_db=data['parameters']['snr_db'],
                                    include_line=noise_types.get('line', True),
                                    include_emg=noise_types.get('emg', True),
                                    include_ecg=noise_types.get('ecg', False),
                                    include_movement=noise_types.get('movement', False),
                                    include_tms=noise_types.get('tms', False)
                                )
                                
                                # Apply filter
                                try:
                                    EP_filtered = filters_obj.apply_filter_cascade(EP_noisy, filter_params)
                                    all_filtered.append(EP_filtered)
                                    
                                    # Plot individual iterations if requested
                                    if show_individual and not show_mean_only:
                                        ax.plot(data['time']*1000, EP_filtered, 
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
                                                   color=color, alpha=0.2, zorder=z_base - 25, label=None)
                                
                                # Plot mean with higher z-order (on top of envelope)
                                ax.plot(data['time']*1000, mean_filtered, 
                                       color=color, linewidth=2.8, alpha=0.95,
                                       label=f'{config_name} (n={len(all_filtered)}) ±SD',
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
                         'correlation', 'rmse_EP', 'baseline_std'],
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
                    # REVIEWER FIX: Use z-scores for correlation
                    heatmap_metric = 'correlation_z' if metric_choice_heat == 'correlation' else metric_choice_heat
                    
                    # Standard sorting for other metrics
                    pivot_data = df.groupby('config_label')[heatmap_metric].mean().sort_values()
                    
                    fig, ax = plt.subplots(figsize=(10, len(pivot_data)*0.3 + 2))
                    
                    # Determine if higher or lower is better
                    if metric_choice_heat in ['correlation']:
                        cmap = 'RdYlGn'  # Higher is better (green)
                        label_text = "Fisher's z-score"
                    else:
                        cmap = 'RdYlGn_r'  # Lower is better (green)
                        label_text = metric_choice_heat
                    
                    sns.heatmap(pivot_data.to_frame(), annot=True, fmt='.3f', 
                               cmap=cmap, ax=ax, cbar_kws={'label': label_text})
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
            
            # Iteration overlay Visualisation
            st.subheader("📊 Filter Performance Visualisation")
            st.write("**Compare filtered signals across iterations with ground truth**")
            
            # Select configuration to visualize
            config_to_plot = st.selectbox(
                "Select filter configuration to visualize:",
                options=df['config_label'].unique(),
                index=0
            )
            
            # Display controls
            col_indiv1, col_indiv2 = st.columns(2)
            with col_indiv1:
                st.write("**Ground Truth Display:**")
                var_enabled_indiv = data['parameters'].get('variability_enabled', False)
                show_gt_var_indiv = st.checkbox("Show ground truth variability", value=var_enabled_indiv, key='show_gt_var_indiv')
                if show_gt_var_indiv and var_enabled_indiv:
                    show_gt_sd_indiv = st.checkbox("Show GT ±SD envelope", value=True, key='show_gt_sd_indiv')
                    show_gt_indiv_lines = st.checkbox("Show individual GTs", value=False, key='show_gt_indiv_lines')
                else:
                    show_gt_sd_indiv = False
                    show_gt_indiv_lines = False
            
            with col_indiv2:
                st.write("**Export:**")
                single_overlay_dpi = st.selectbox("Export DPI", [150, 300, 600], index=1, key='single_overlay_dpi')
            
            # Add DPI selector
            
            if st.button("🎨 Generate Overlay Plot"):
                with st.spinner("Generating Visualisation..."):
                    # Initialize generators and filters for this Visualisation
                    noise_gen = NoiseGenerator(sampling_rate=data['sampling_rate'])
                    filters = EPFilters(sampling_rate=data['sampling_rate'])
                    
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
                        'notch_freq': data['parameters'].get('line_freq', 50)
                    }
                    
                    # Collect ground truths and apply filters
                    all_gt_indiv = []
                    all_filtered_indiv = []
                    
                    # Apply filter and plot each iteration (thin lines)
                    for idx, row in config_data.iterrows():
                        # Regenerate EP with same seed as batch
                        if data['parameters'].get('variability_enabled', False):
                            import numpy as np
                            from signal_generator import EPGenerator
                            tg_indiv = EPGenerator(sampling_rate=data['sampling_rate'])
                            np.random.seed(row.get('seed_used', row['iteration']))
                            
                            a_i, l_i, d_i = tg_indiv.apply_EP_variability(
                                data['parameters']['amplitude'],
                                data['parameters']['onset_latency']/1000,
                                data['parameters']['duration']/1000,
                                data['parameters']['amplitude_variability'],
                                data['parameters']['latency_variability']/1000,
                                data['parameters']['duration_variability']
                            )
                            
                            EP_type_i = data['parameters'].get('EP_type', 'Standard')
                            if EP_type_i == 'Bi-phasic':
                                time_i, EP_clean_i = tg_indiv.generate_biphasic_EP(a_i * (-1 if data['parameters'].get('invert_polarity', False) else 1), l_i,
                                    data['parameters'].get('rise_time', 8)/1000*1.2,
                                    data['parameters'].get('decay_time', 15)/1000, 0.8,
                                    include_onset_dip=data['parameters'].get('include_onset_dip', True))
                            elif EP_type_i == 'Tri-phasic':
                                time_i, EP_clean_i = tg_indiv.generate_triphasic_EP(a_i * (-1 if data['parameters'].get('invert_polarity', False) else 1), l_i,
                                    data['parameters'].get('rise_time', 8)/1000,
                                    data['parameters'].get('decay_time', 15)/1000*1.2,
                                    data['parameters'].get('rise_time', 8)/1000*1.3, 0.75, 0.4,
                                    include_onset_dip=data['parameters'].get('include_onset_dip', True))
                            else:
                                time_i, EP_clean_i = tg_indiv.generate_EP(a_i, d_i, l_i,
                                    data['parameters'].get('rise_time', 8)/1000,
                                    data['parameters'].get('decay_time', 15)/1000)
                        else:
                            time_i = data['time']
                            EP_clean_i = data['EP_clean']
                        
                        all_gt_indiv.append(EP_clean_i)
                        
                        # Generate noisy signal for this iteration
                        # CRITICAL FIX: Use actual noise settings from GUI, not hardcoded values
                        noise_types = data['parameters'].get('noise_types', {})
                        EP_noisy = noise_gen.add_composite_noise(
                            EP_clean_i, time_i,
                            snr_db=data['parameters']['snr_db'],
                            include_line=noise_types.get('line', True),
                            include_emg=noise_types.get('emg', True),
                            include_ecg=noise_types.get('ecg', False),
                            include_movement=noise_types.get('movement', False),
                            include_tms=noise_types.get('tms', False)
                        )
                        
                        # Apply filter
                        try:
                            EP_filtered = filters.apply_filter_cascade(EP_noisy, filter_params)
                            
                            # Plot filtered iteration (thin, semi-transparent)
                            axes[0].plot(data['time']*1000, EP_filtered, 'r-', 
                                       linewidth=0.5, alpha=0.3, zorder=1)
                            all_filtered_indiv.append(EP_filtered)
                        except:
                            pass  # Skip if filter fails
                    
                    # Plot mean ground truth with variability
                    if len(all_gt_indiv) > 0:
                        mean_gt_indiv = np.mean(all_gt_indiv, axis=0)
                        
                        # Show individual GT lines on TOP plot if enabled
                        if show_gt_var_indiv and data['parameters'].get('variability_enabled', False):
                            if show_gt_indiv_lines:
                                for gt in all_gt_indiv:
                                    axes[0].plot(data['time']*1000, gt, 'b-', lw=0.3, alpha=0.15, zorder=50)
                                # Add legend entry for individual GTs
                                axes[0].plot([], [], 'b-', lw=0.3, alpha=0.4, label=f'Individual GTs (n={len(all_gt_indiv)})')
                    else:
                        mean_gt_indiv = data['EP_clean']
                    
                    # Plot mean ground truth on top plot (bold, no envelope yet)
                    axes[0].plot(data['time']*1000, mean_gt_indiv, 'b-', 
                               linewidth=2.5, alpha=1.0, label='Ground Truth (Mean)', zorder=100)
                    
                    # Add single legend entry for iterations
                    axes[0].plot([], [], 'r-', linewidth=0.5, alpha=0.5, label=f'Filtered (n={len(config_data)})')
                    
                    # Add stimulus marker
                    axes[0].axvline(0, color='k', linestyle='--', alpha=0.4, linewidth=1.5, label='TMS Stimulus')
                    
                    axes[0].set_ylabel('Amplitude (mV)', fontsize=12)
                    axes[0].set_title(f'Overlay: {config_to_plot}', fontsize=14, fontweight='bold')
                    axes[0].legend(loc='upper right', fontsize=11)
                    axes[0].grid(True, alpha=0.3)
                    
                    # Bottom plot: Mean ± SD envelope
                    # Reuse data collected in top plot
                    all_filtered = all_filtered_indiv
                    all_gt_mean_sd = all_gt_indiv
                    
                    if len(all_filtered) > 0:
                        all_filtered = np.array(all_filtered)
                        mean_filtered = np.mean(all_filtered, axis=0)
                        std_filtered = np.std(all_filtered, axis=0)
                        
                        # Calculate mean ground truth
                        if len(all_gt_mean_sd) > 0:
                            mean_gt_sd = np.mean(all_gt_mean_sd, axis=0)
                        else:
                            mean_gt_sd = data['EP_clean']
                        
                        # Plot ground truth with ±SD envelope if variability enabled
                        axes[1].plot(data['time']*1000, mean_gt_sd, 'b-', 
                                   linewidth=2.5, alpha=0.8, label='Ground Truth ±SD')
                        
                        # Add GT ±SD envelope if enabled
                        if show_gt_var_indiv and show_gt_sd_indiv and len(all_gt_mean_sd) > 1:
                            std_gt_sd = np.std(all_gt_mean_sd, axis=0)
                            axes[1].fill_between(data['time']*1000,
                                               mean_gt_sd - std_gt_sd,
                                               mean_gt_sd + std_gt_sd,
                                               color='blue', alpha=0.15)
                        
                        # Plot mean filtered with ±SD envelope
                        axes[1].plot(data['time']*1000, mean_filtered, 'r-', 
                                   linewidth=2, alpha=0.8, label='Mean Filtered ±SD')
                        axes[1].fill_between(data['time']*1000, 
                                           mean_filtered - std_filtered,
                                           mean_filtered + std_filtered,
                                           color='r', alpha=0.2)
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
                            'notch_freq': data['parameters'].get('line_freq', 50)
                        }
                        
                        # Initialize for this specific analysis
                        noise_gen_tf = NoiseGenerator(sampling_rate=data['sampling_rate'])
                        filters_tf = EPFilters(sampling_rate=data['sampling_rate'])
                        
                        # Generate noisy signal for this iteration
                        np.random.seed(tf_iteration)  # Reproducible for same iteration
                        # CRITICAL FIX: Use actual noise settings from GUI, not hardcoded values
                        noise_types = data['parameters'].get('noise_types', {})
                        EP_noisy_tf = noise_gen_tf.add_composite_noise(
                            data['EP_clean'], data['time'],
                            snr_db=data['parameters']['snr_db'],
                            include_line=noise_types.get('line', True),
                            include_emg=noise_types.get('emg', True),
                            include_ecg=noise_types.get('ecg', False),
                            include_movement=noise_types.get('movement', False),
                            include_tms=noise_types.get('tms', False)
                        )
                        
                        # Apply filter
                        EP_filtered_tf = filters_tf.apply_filter_cascade(EP_noisy_tf, filter_params)
                        
                        # Generate time-frequency plot
                        tf_fig = plot_timefreq_comparison(
                            data['EP_clean'],
                            EP_noisy_tf,
                            EP_filtered_tf,
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
                            - Shows natural EP frequency content
                            - Typically concentrated in 20-150 Hz range
                            - Transient burst at EP onset
                            - Should be temporally localized (15-60 ms)
                            
                            **Panel B - Noisy Signal:**
                            - Broadband noise across all frequencies
                            - Line noise appears as horizontal streak (50 Hz, 100 Hz)
                            - TMS Artefact: high-frequency burst at t=0
                            - EP signal buried in noise
                            
                            **Panel C - Filtered Signal:**
                            - Should match Panel A in EP window
                            - Noise frequencies attenuated
                            - Assess: Is EP spectral structure preserved?
                            - Check: Any spurious frequencies introduced?
                            
                            ### Good Filter Characteristics
                            
                            ✓ Preserves EP frequency content (20-150 Hz in EP window)
                            ✓ Removes high-frequency noise (>300 Hz)
                            ✓ Removes low-frequency drift (<10 Hz)
                            ✓ No introduction of spurious frequencies
                            ✓ Maintains temporal localization
                            
                            ### Red Flags
                            
                            ✗ EP frequency content reduced/distorted
                            ✗ New frequencies appear (filter ringing)
                            ✗ Temporal smearing (energy spreads in time)
                            ✗ Asymmetric frequency attenuation
                            """)
                    else:
                        st.error(f"Iteration {tf_iteration} not available for this configuration")
            
            # Download results
            st.subheader("📥 Export Results")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="EP_filter_batch_results.csv",
                    mime="text/csv"
                )
            
            with col_dl2:
                # Download reproducibility parameters
                import json
                
                # Build iteration details with exact parameters
                iter_details = []
                for _, row in df.iterrows():
                    iter_dict = {
                        "iteration": int(row["iteration"]),
                        "config_label": row["config_label"],
                        "seed_used": int(row.get("seed_used", 0))
                    }
                    if "actual_amplitude" in row:
                        iter_dict["actual_amplitude"] = float(row["actual_amplitude"])
                        iter_dict["actual_latency"] = float(row["actual_latency"])
                        iter_dict["actual_duration"] = float(row["actual_duration"])
                    iter_details.append(iter_dict)
                
                repro_params = {
                    "master_seed": st.session_state.get('master_seed_used', None),
                    "signal_parameters": data["parameters"],
                    "iteration_parameters": iter_details,
                    "tool_version": "1.0.0",
                    "timestamp": str(pd.Timestamp.now())
                }
                
                repro_json = json.dumps(repro_params, indent=2, default=str)
                st.download_button(
                    label="🔢 Download Reproducibility Params (JSON)",
                    data=repro_json,
                    file_name="EP_reproducibility_params.json",
                    mime="application/json",
                    help="Exact parameters for each iteration"
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
            st.metric("EP Type", params.get('EP_type', 'Not set'))
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
        methods_parts.append("Evoked potential simulations and digital filter evaluations were performed using the ")
        methods_parts.append("EPSimFilt version 1.0.0 (Andrushko, 2025), implemented in Python 3.8+. ")
        
        if detail_level != "Standard (Journal Article)":
            methods_parts.append("Core computational dependencies included: NumPy 1.24+ (array operations and numerical computing), ")
            methods_parts.append("SciPy 1.11+ (signal processing and filter design via scipy.signal module), ")
            methods_parts.append("Matplotlib 3.7+ (visualisation), Pandas 2.0+ (data management and aggregation), ")
            methods_parts.append("and Seaborn 0.12+ (statistical graphics). ")
        
        if detail_level == "Complete (Technical Documentation)":
            methods_parts.append("All numerical operations were performed using double-precision floating-point ")
            methods_parts.append("arithmetic (float64) to ensure numerical stability. ")
        
        methods_parts.append("\n\n## Signal Generation\n\n")
        
        # EP morphology description with COMPLETE parameters
        EP_type = params.get('EP_type', 'Standard')
        amplitude = params.get('amplitude', 1.0)
        duration = params.get('duration', 30)
        onset_latency = params.get('onset_latency', 20)
        rise_time = params.get('rise_time', 8)
        decay_time = params.get('decay_time', 15)
        asymmetry = params.get('asymmetry', 1.2)
        
        if EP_type == "Mono-Phasic":
            methods_parts.append(f"Monophasic evoked potentials were generated using gamma distribution-based ")
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
        
        elif EP_type == "Bi-phasic":
            methods_parts.append(f"Bi-phasic evoked potentials (positive-negative morphology) were generated ")
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
        
        elif EP_type == "Tri-phasic":
            methods_parts.append(f"Tri-phasic evoked potentials (positive-negative-positive morphology) ")
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
                    methods_parts.append("Noise power was calculated relative to signal power in the EP window ")
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
                    methods_parts.append("preserving evoked potential latency measurements. ")
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
                    methods_parts.append("Selected configurations included a notch filter for line noise rejection. ")
                else:
                    methods_parts.append("Selected configurations incorporated a notch filter ")
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
                methods_parts.append("peak-to-peak amplitude from ground truth within the EP window ")
                methods_parts.append("(15-80 ms post-stimulus): E_amp = 100 × (A_filt - A_true) / A_true.\n\n")
                
                methods_parts.append("**Peak Latency Shift:** Temporal displacement (milliseconds) ")
                methods_parts.append("of maximum absolute amplitude.\n\n")
                
                methods_parts.append("**Pearson Correlation:** Computed via np.corrcoef() between ")
                methods_parts.append("filtered and ground truth waveforms for morphological fidelity.\n\n")
                
                methods_parts.append("**RMSE:** Root mean square error calculated over EP window.\n\n")
                
                methods_parts.append("**SNR Improvement:** Post-filtering SNR minus pre-filtering SNR (dB). ")
            
            else:  # Complete Technical Documentation
                methods_parts.append("**Amplitude Preservation Error:**\n\n")
                methods_parts.append("E_amp = 100 × (A_filtered - A_true) / A_true (%)\n\n")
                methods_parts.append("where A = max(x) - min(x) within EP window (15-80 ms).\n\n")
                
                methods_parts.append("**Onset Latency Error:**\n\n")
                methods_parts.append("Onset was detected using a robust peak-first backwards search algorithm. ")
                methods_parts.append("The method first identified the maximum deflection (peak) in the 15-80 ms post-stimulus window, ")
                methods_parts.append("then worked backwards to identify where the signal first deviated from baseline ")
                methods_parts.append("(mean ± 2σ, baseline: 0-15 ms). ")
                methods_parts.append("This approach is more robust to noise and artefacts than traditional threshold-crossing methods ")
                methods_parts.append("(Rossini et al., 2015). ")
                methods_parts.append("Error = onset_filtered - onset_true (ms).\n\n")
                references_used.append('rossini2015')
                
                methods_parts.append("**Peak Latency Error:**\n\n")
                methods_parts.append("E_latency = argmax(|x_filtered|) - argmax(|x_true|) (ms)\n\n")
                
                methods_parts.append("**Pearson Correlation Coefficient:**\n\n")
                methods_parts.append("r = Σ[(x_i-μ_x)(y_i-μ_y)] / √[Σ(x_i-μ_x)²·Σ(y_i-μ_y)²]\n\n")
                methods_parts.append("Fisher's z-transformation was applied for statistical analysis: ")
                methods_parts.append("z = 0.5 × ln[(1+r)/(1-r)] = arctanh(r). ")
                methods_parts.append("This transformation is required because correlation coefficients have non-normal ")
                methods_parts.append("sampling distributions (Fisher, 1915).\n\n")
                references_used.append('fisher1915')
                
                methods_parts.append("**Root Mean Square Error:**\n\n")
                methods_parts.append("RMSE = √[(1/N)Σ(x_filt[i] - x_true[i])²] (mV)\n\n")
                methods_parts.append("Calculated over: (1) full signal, (2) EP window only.\n\n")
                
                methods_parts.append("**Signal-to-Noise Ratio:**\n\n")
                methods_parts.append("SNR_dB = 10·log₁₀(P_signal/P_noise)\n\n")
                methods_parts.append("where P = mean(x²) in respective windows.\n\n")
                
                methods_parts.append("**Baseline Stability:** σ of filtered signal in 0-15 ms (lower = better).\n\n")
                
                methods_parts.append("**Spurious Oscillations:** Detected if baseline envelope >5σ ")
                methods_parts.append("or zero-crossing rate >30% (indicates filter ringing). ")
            
            references_used.append('groppa2012')
            references_used.append('rossini2015')
            
            methods_parts.append("\n\n## Data Analysis and Statistical Testing\n\n")
            
            if detail_level == "Standard (Journal Article)":
                methods_parts.append(f"Performance metrics were aggregated across {n_iterations} iterations ")
                methods_parts.append("using mean and standard deviation. ")
                methods_parts.append("Prior to statistical testing, data distributions were assessed for normality ")
                methods_parts.append("(Shapiro-Wilk test) and bimodality (bimodality coefficient). ")
                methods_parts.append("Metrics exhibiting positive skewness and non-normality were log-transformed. ")
                methods_parts.append("Statistical comparisons between filter configurations were performed using ")
                methods_parts.append("[specify your chosen test: one-way ANOVA or Kruskal-Wallis H-test] ")
                methods_parts.append("with appropriate post-hoc tests and corrections for multiple comparisons. ")
                methods_parts.append("Statistical significance was set at α = 0.05.\n\n")
            
            elif detail_level == "Detailed (Methods Paper)":
                methods_parts.append(f"**Data Aggregation:** Each filter configuration's performance was assessed across ")
                methods_parts.append(f"{n_iterations} independent noise realisations. ")
                methods_parts.append("Metrics were aggregated using arithmetic mean (μ = Σx_i / n) ")
                methods_parts.append("and sample standard deviation [σ = √(Σ(x_i-μ)² / (n-1))]. ")
                methods_parts.append("95% confidence intervals were calculated using the t-distribution: ")
                methods_parts.append("CI = μ ± t_(α/2,n-1) · (σ/√n).\n\n")
                
                methods_parts.append("**Assumption Testing:** Prior to statistical analysis, data distributions were evaluated for:\n\n")
                methods_parts.append("- **Normality**: Shapiro-Wilk test (H₀: data follows normal distribution, α = 0.05)\n")
                methods_parts.append("- **Bimodality**: Bimodality coefficient BC = (g₁² + 1) / g₂ where g₁ = skewness, g₂ = kurtosis. ")
                methods_parts.append("BC > 0.555 indicates bimodal distribution\n")
                methods_parts.append("- **Homogeneity of variance**: Levene's test for equality of variances across groups\n\n")
                
                methods_parts.append("**Data Transformations:** Metrics exhibiting positive skewness (|g₁| > 1) and ")
                methods_parts.append("non-normality (Shapiro p < 0.05) were log-transformed using a sign-preserving transformation: ")
                methods_parts.append("x_log = sign(x) · ln(|x| + ε) where ε = 10⁻¹⁰ prevents ln(0). ")
                methods_parts.append("Both raw and log-transformed metrics were exported for statistical analysis. ")
                methods_parts.append("Fisher's z-transformation [z = arctanh(r)] was applied to correlation coefficients ")
                methods_parts.append("prior to averaging and statistical testing (Fisher, 1915).\n\n")
                references_used.append('fisher1915')
                
                methods_parts.append("**Statistical Testing:** [Describe your specific tests here based on assumption testing results. ")
                methods_parts.append("Example: \"One-way ANOVA was used for normally distributed metrics (amplitude error, peak latency error) ")
                methods_parts.append("with Tukey's HSD post-hoc test. Kruskal-Wallis test was used for non-normally distributed metrics ")
                methods_parts.append("with Dunn's post-hoc test and Benjamini-Hochberg FDR correction. ")
                methods_parts.append("Effect sizes were quantified using Cohen's d (parametric) or rank-biserial correlation (non-parametric).\"]\n\n")
            
            else:  # Complete (Technical Documentation)
                methods_parts.append(f"**Aggregation Across Iterations ({n_iterations} per configuration):**\n\n")
                methods_parts.append("Descriptive Statistics:\n")
                methods_parts.append("- Mean: μ = (1/n)Σx_i\n")
                methods_parts.append("- Standard Deviation: σ = √[(1/(n-1))Σ(x_i-μ)²]\n")
                methods_parts.append("- Median: 50th percentile via NumPy's median function\n")
                methods_parts.append("- 95% Confidence Interval: CI = μ ± t_(α/2,n-1) · SE where SE = σ/√n\n\n")
                
                methods_parts.append("**Distribution Analysis:**\n\n")
                methods_parts.append("Normality Assessment:\n")
                methods_parts.append("- Shapiro-Wilk test: W = (Σa_i·x_(i))² / Σ(x_i-x̄)² where a_i are tabulated coefficients\n")
                methods_parts.append("- Skewness: g₁ = [n/((n-1)(n-2))]Σ[(x_i-μ)/σ]³\n")
                methods_parts.append("- Excess kurtosis: g₂ = [n(n+1)/((n-1)(n-2)(n-3))]Σ[(x_i-μ)/σ]⁴ - 3(n-1)²/((n-2)(n-3))\n\n")
                
                methods_parts.append("Bimodality Coefficient:\n")
                methods_parts.append("BC = (g₁² + 1) / g₂\n")
                methods_parts.append("Interpretation: BC > 5/9 ≈ 0.555 suggests multimodality\n\n")
                
                methods_parts.append("**Data Transformations:**\n\n")
                methods_parts.append("Log Transformation (sign-preserving):\n")
                methods_parts.append("x_log = sign(x) · ln(|x| + ε) where ε = 10⁻¹⁰\n\n")
                methods_parts.append("Fisher's z-transformation for correlation coefficients:\n")
                methods_parts.append("z = 0.5 · ln[(1+r)/(1-r)] = arctanh(r)\n")
                methods_parts.append("Inverse: r = tanh(z) = (e²ᶻ - 1)/(e²ᶻ + 1)\n\n")
                
                methods_parts.append("**Statistical Testing:** All raw and transformed metrics were exported to CSV format ")
                methods_parts.append("for statistical analysis in [specify software]. Test selection was guided by assumption ")
                methods_parts.append("testing results. For normally distributed metrics, parametric tests (one-way ANOVA with ")
                methods_parts.append("Tukey HSD or independent t-tests) were used. For non-normal metrics, non-parametric ")
                methods_parts.append("alternatives (Kruskal-Wallis H-test with Dunn's post-hoc or Mann-Whitney U) were employed. ")
                methods_parts.append("Multiple comparison corrections (Bonferroni or Benjamini-Hochberg FDR) were applied as appropriate. ")
                methods_parts.append("Effect sizes were calculated as Cohen's d for parametric tests or rank-biserial correlation ")
                methods_parts.append("for non-parametric tests.\n\n")
            
            if detail_level != "Standard (Journal Article)":
                methods_parts.append("**Visualisation Methods:**\n\n")
                methods_parts.append(f"- Box plots: Median, IQR, whiskers to 1.5×IQR (n={n_iterations} per box)\n")
                methods_parts.append("- Heatmaps: Mean values, diverging colour map centred at zero\n")
                methods_parts.append("- Overlay plots: Individual iterations + mean ± 1σ envelope\n\n")
        
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
            full_methods += "Andrushko, J.W. (2025). EPSimFilt:  "
            full_methods += "A Systematic Digital Filter Evaluation tool for Evoked Potentials (Version 1.0.0). Northumbria University. "
            full_methods += "https://github.com/jandrushko/EPSimFilt\n\n"
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
                file_name=f"EP_methods_{detail_level.split()[0].lower()}_{EP_type.lower().replace('-','')}.txt",
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

TY  - JOUR
AU  - Fisher, Ronald A
TI  - Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population
JO  - Biometrika
VL  - 10
IS  - 4
SP  - 507
EP  - 521
PY  - 1915
DO  - 10.2307/2331838
ER  -

TY  - SOFT
AU  - Andrushko, Justin W
TI  - EPSimFilt: A Systematic Digital Filter Evaluation tool for Evoked Potentials
PY  - 2025
PB  - Northumbria University
VL  - 1.0
UR  - https://github.com/jandrushko/EPSimFilt
ER  -

"""
            
            st.download_button(
                label="📚 Download References (RIS)",
                data=ris_content,
                file_name="EP_filter_references.ris",
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
        
        3. **Andrushko, J.W. (2025).** EPSimFilt: A Systematic Digital Filter Evaluation tool for Evoked Potentials (Version 1.0.0). Northumbria University. 
           https://github.com/jandrushko/EPSimFilt
        
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
    ### EPSimFilt: EP Filter Testing Tool
    
    **Version:** 1.0.0  
    **Released:** December 2025  
    **Author:** Justin W. Andrushko, PhD  
    **Institution:** Northumbria University
    
    ### Purpose
    
    This tool was developed to systematically evaluate digital filter performance for 
    Evoked Potential (EP) analysis in transcranial magnetic stimulation (TMS) studies. 
    It addresses the critical need for evidence-based filter selection through rigorous 
    simulation-based testing with comprehensive statistical analysis.
    
    ### Key Features
    
    **Signal Generation:**
    - Multiple EP morphologies (monophasic, bi-phasic, tri-phasic)
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
    
    Andrushko, J.W. (2025). EPSimFilt: A Systematic Digital Filter Evaluation tool for Evoked Potentials (Version 1.0.0). Northumbria University. 
    https://github.com/jandrushko/EPSimFilt/
    
    **TMSMultiLab:**
    
    Part of the TMSMultiLab initiative for advancing TMS methodological standards.
    Visit: https://github.com/TMSMultiLab/TMSMultiLab/wiki
    
    ### Recommended Workflow
    
    1. **Generate Signal**: Create realistic EP with desired noise characteristics
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
    Evoked Potential (EP) analysis in TMS studies. It addresses the critical need 
    for evidence-based filter selection in neurophysiology research.
    
    ### Features
    
    - **Realistic EP Simulation**: Generate EPs with configurable amplitude, duration, 
      rise/fall times, and morphology
    - **Comprehensive Noise Models**: Add physiological (EMG, ECG) and technical 
      (line noise, amplifier noise) artefacts
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
    - EP onset latency error
    - Peak latency error
    
    **Morphology Metrics:**
    - Correlation with ground truth
    - RMSE (full signal and EP window)
    - Baseline stability
    
    **Quality Metrics:**
    - Signal-to-noise ratio improvement
    - Spurious oscillation detection
    
    ### Recommended Workflow
    
    1. **Generate Signal**: Start with the Signal Generation tab to create a realistic EP 
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
    Andrushko, J.W. (2025). EPSimFilt: A Systematic Digital Filter Evaluation tool for Evoked Potentials.
    Northumbria University.
    ```
    
    ### Feedback & Contributions
    
    This tool is open for community feedback and contributions. If you have suggestions 
    for improvements, bug reports, or feature requests, please contact the author.
    
    ### Acknowledgments
    
    Developed with support from the TMSMultiLab and Northumbria University's Neuromuscular Function research group.
    
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