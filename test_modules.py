"""
Test Script: Verify all modules work correctly
Run this before using the tool to ensure everything is properly installed.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from signal_generator import MEPGenerator
        from noise_generator import NoiseGenerator
        from filters import MEPFilters
        from metrics import MEPMetrics
        print("  ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_signal_generation():
    """Test MEP signal generation."""
    print("\nTesting signal generation...")
    try:
        from signal_generator import MEPGenerator
        
        gen = MEPGenerator(sampling_rate=2000)
        time, mep = gen.generate_mep(amplitude=1.0)
        
        assert len(time) == len(mep), "Time and MEP arrays must have same length"
        assert time[-1] > 0, "Time vector must be positive"
        assert mep.max() > 0, "MEP must have positive values"
        
        print("  ✓ Signal generation works")
        return True
    except Exception as e:
        print(f"  ✗ Signal generation error: {e}")
        return False

def test_noise_addition():
    """Test noise addition."""
    print("\nTesting noise addition...")
    try:
        from signal_generator import MEPGenerator
        from noise_generator import NoiseGenerator
        
        gen = MEPGenerator(sampling_rate=2000)
        noise_gen = NoiseGenerator(sampling_rate=2000)
        
        time, mep = gen.generate_mep(amplitude=1.0)
        mep_noisy = noise_gen.add_white_noise(mep, snr_db=10.0)
        
        assert len(mep_noisy) == len(mep), "Noisy signal must have same length"
        
        print("  ✓ Noise addition works")
        return True
    except Exception as e:
        print(f"  ✗ Noise addition error: {e}")
        return False

def test_filtering():
    """Test filter application."""
    print("\nTesting filtering...")
    try:
        from signal_generator import MEPGenerator
        from filters import MEPFilters
        
        gen = MEPGenerator(sampling_rate=2000)
        filters = MEPFilters(sampling_rate=2000)
        
        time, mep = gen.generate_mep(amplitude=1.0)
        
        filter_params = {
            'filter_type': 'butterworth',
            'lowcut': 10,
            'highcut': 500,
            'order': 4,
            'notch_enabled': False
        }
        
        mep_filtered = filters.apply_filter_cascade(mep, filter_params)
        
        assert len(mep_filtered) == len(mep), "Filtered signal must have same length"
        
        print("  ✓ Filtering works")
        return True
    except Exception as e:
        print(f"  ✗ Filtering error: {e}")
        return False

def test_metrics():
    """Test metrics calculation."""
    print("\nTesting metrics calculation...")
    try:
        from signal_generator import MEPGenerator
        from noise_generator import NoiseGenerator
        from filters import MEPFilters
        from metrics import MEPMetrics
        
        gen = MEPGenerator(sampling_rate=2000)
        noise_gen = NoiseGenerator(sampling_rate=2000)
        filters = MEPFilters(sampling_rate=2000)
        metrics_calc = MEPMetrics(sampling_rate=2000)
        
        time, mep_clean = gen.generate_mep(amplitude=1.0)
        mep_noisy = noise_gen.add_white_noise(mep_clean, snr_db=10.0)
        
        filter_params = {
            'filter_type': 'butterworth',
            'lowcut': 10,
            'highcut': 500,
            'order': 4,
            'notch_enabled': False
        }
        
        mep_filtered = filters.apply_filter_cascade(mep_noisy, filter_params)
        metrics = metrics_calc.calculate_all_metrics(mep_clean, mep_filtered, time)
        
        assert 'amplitude_error_pct' in metrics, "Metrics must include amplitude error"
        assert 'correlation' in metrics, "Metrics must include correlation"
        
        print("  ✓ Metrics calculation works")
        print(f"     - Amplitude error: {metrics['amplitude_error_pct']:.2f}%")
        print(f"     - Correlation: {metrics['correlation']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Metrics calculation error: {e}")
        return False

def test_dependencies():
    """Test that all required packages are installed."""
    print("\nTesting dependencies...")
    required_packages = [
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'streamlit'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not installed")
            all_installed = False
    
    return all_installed

def main():
    """Run all tests."""
    print("=" * 60)
    print("MEP Filter Tool - Module Verification")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Signal Generation", test_signal_generation),
        ("Noise Addition", test_noise_addition),
        ("Filtering", test_filtering),
        ("Metrics", test_metrics)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! The tool is ready to use.")
        print("\nTo launch the GUI, run: streamlit run app.py")
        print("Or use the launcher scripts: ./launch.sh (Mac/Linux) or launch.bat (Windows)")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Python path issues: Ensure src/ directory exists")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
