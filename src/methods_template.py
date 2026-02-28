"""
Generate adaptive methods section based on user settings
"""

def generate_methods_text(settings):
    """Generate methods text based on analysis settings."""
    
    methods = []
    
    # Signal generation
    if settings.get('EP_type'):
        morphology = settings['EP_type']
        amplitude = settings.get('amplitude', 1.0)
        duration = settings.get('duration', 100)
        sampling_rate = settings.get('sampling_rate', 5000)
        
        methods.append(f"**Signal Generation**\n\n")
        methods.append(f"{morphology} evoked potentials were simulated using the EP Filter Testing Tool v1.4.0 (Andrushko, 2024) ")
        methods.append(f"with amplitude {amplitude} mV and total duration {duration} ms, sampled at {sampling_rate} Hz. ")
    
    # Noise description
    if settings.get('snr_db'):
        snr = settings['snr_db']
        noise_types = settings.get('noise_types', [])
        methods.append(f"Realistic physiological noise (signal-to-noise ratio: {snr} dB) was added, including ")
        
        noise_desc = []
        if 'emg' in noise_types:
            noise_desc.append("electromyographic activity (20-500 Hz)")
        if 'line' in noise_types:
            noise_desc.append("50 Hz line noise with harmonics")
        if 'ecg' in noise_types:
            noise_desc.append("electrocardiographic artefacts")
        if 'tms' in noise_types:
            noise_desc.append("residual transcranial magnetic stimulation artefact (1.5 mV, 2 ms decay)")
        
        methods.append(", ".join(noise_desc) + ". ")
    
    # Filter testing
    if settings.get('filter_configs'):
        configs = settings['filter_configs']
        iterations = settings.get('iterations', 10)
        
        methods.append(f"\n\n**Filter Evaluation**\n\n")
        methods.append(f"Digital filters were evaluated systematically across {len(configs)} configurations ")
        methods.append(f"tested across {iterations} noise realisations. ")
        methods.append("Filter performance was quantified using amplitude preservation error (%), ")
        methods.append("peak latency shift (ms), and Pearson correlation coefficient with ground truth waveforms. ")
        methods.append("Results are reported as mean Â± standard deviation across all iterations.")
    
    return "".join(methods)

def generate_ris_references():
    """Generate RIS formatted references."""
    
    ris = """TY  - JOUR
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
AU  - Valls-SolÃ©, Josep
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
TI  - EP Filter Testing Tool: Systematic Digital Filter Evaluation for Evoked Potentials
PY  - 2024
PB  - Northumbria University
VL  - 1.4.0
UR  - https://github.com/andrushko/EP-filter-tool
ER  -
"""
    
    return ris
