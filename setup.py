import os
from pathlib import Path

class Setup:
    # Audio settings
    SAMPLE_RATE = 16000
    BATCH_SIZE = 12000
    
    # Model settings
    MODEL_PATH = os.path.join("model", "nocle.hdf5")
    MODEL_OPTIMIZER = 'adam'
    MODEL_LOSS = 'mse'
    
    # Filter parameters
    WIENER_FILTER_SIZE = 15
    WIENER_FILTER_NOISE_VAR = 0.01
    WIENER_SIZE_RANGE = (3, 31, 2)  # (min, max, step)
    
    GAUSSIAN_BLUR_SIGMA = 2.0
    GAUSSIAN_SIGMA_RANGE = (0.1, 5.0, 0.1)  # (min, max, step)
    
    NOISE_GATE_THRESHOLD = 0.01
    
    DYNAMIC_EXPANSION_THRESHOLD = 0.35
    DYNAMIC_EXPANSION_RATIO = 1.5
    
    EXPONENTIAL_SMOOTH_ALPHA = 0.9
    
    # Filter default states
    DEFAULT_SPECTRAL_GATE = False
    DEFAULT_WIENER = False
    DEFAULT_GAUSSIAN = False
    DEFAULT_SHOW_SPECTROGRAMS = False
    
    # Spectral gate parameters
    SPECTRAL_GATE_THRESHOLD = 1.5  # Multiplier for noise threshold
    SPECTRAL_GATE_N_FFT = 2048
    SPECTRAL_GATE_HOP_LENGTH = 512
    
    # Window dimensions
    MAIN_WINDOW_SIZE = "800x700"
    SPECTROGRAM_WINDOW_SIZE = "1000x700"
    
    # Temporary files
    TEMP_PROCESSED_AUDIO = "temp_processed.wav"
    
    # GUI text
    WINDOW_TITLE = "Nocle Audio Processing"
    READY_STATUS = "Ready"
    PROCESSING_STATUS = "Processing audio..."
    PROCESSING_COMPLETE = "Processing completed successfully"
    PROCESSING_FAILED = "Processing failed"
    
    # File dialog settings
    FILE_TYPES = [
        ("WAV files", "*.wav"),
        ("All files", "*.*")
    ]
    
    # Error messages
    ERROR_NO_FILE = "Please select an audio file first"
    ERROR_NO_PROCESSED = "Please process an audio file first"
    ERROR_MODEL_LOAD = "Failed to load model"
    
    # Success messages
    SUCCESS_MODEL_LOAD = "Model loaded successfully"
    SUCCESS_PROCESSING = "Audio processing completed"

    @staticmethod
    def get_model_path():
        """Get the absolute path to the model file"""
        return os.path.abspath(Setup.MODEL_PATH)
    
    @staticmethod
    def format_time(seconds):
        """Format seconds into MM:SS"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}:{seconds:02d}"
    
    @staticmethod
    def format_time_label(current, total):
        """Format time label as MM:SS / MM:SS"""
        return f"{Setup.format_time(current)} / {Setup.format_time(total)}"