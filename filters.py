import numpy as np
import librosa
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter1d

from setup import Setup

class AudioFilters:
    @staticmethod
    def noise_gate(data, threshold=Setup.NOISE_GATE_THRESHOLD):
        """Apply noise gate filter"""
        return np.where(np.abs(data) > threshold, data, 0)

    @staticmethod
    def dynamic_expansion(data, threshold=Setup.DYNAMIC_EXPANSION_THRESHOLD, ratio=Setup.DYNAMIC_EXPANSION_RATIO):
        """Apply dynamic expansion"""
        expanded = np.where(
            np.abs(data) > threshold,
            np.sign(data) * (np.abs(data) ** ratio),
            data
        )
        return expanded / np.max(np.abs(expanded))

    @staticmethod
    def exponential_smooth(data, alpha=Setup.EXPONENTIAL_SMOOTH_ALPHA):
        """Apply exponential smoothing"""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for t in range(1, len(data)):
            smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
        return smoothed

    @staticmethod
    def spectral_gating(noisy_signal, sr):
        """Apply spectral gating for noise reduction"""
        stft = librosa.stft(noisy_signal, 
                           n_fft=Setup.SPECTRAL_GATE_N_FFT, 
                           hop_length=Setup.SPECTRAL_GATE_HOP_LENGTH)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        noise_thresh = np.median(magnitude, axis=1)[:, None]
        mask = magnitude > (Setup.SPECTRAL_GATE_THRESHOLD * noise_thresh)
        
        filtered_stft = stft * mask
        return librosa.istft(filtered_stft, hop_length=Setup.SPECTRAL_GATE_HOP_LENGTH)

    @staticmethod
    def wiener_filter(audio, mysize=Setup.WIENER_FILTER_SIZE, noise_var=Setup.WIENER_FILTER_NOISE_VAR):
        """Apply Wiener filter"""
        return wiener(audio, mysize=mysize, noise=noise_var)

    @staticmethod
    def gaussian_blur(audio, sigma=Setup.GAUSSIAN_BLUR_SIGMA):
        """Apply Gaussian blur"""
        return gaussian_filter1d(audio, sigma=sigma)

    @classmethod
    def apply_all_filters(cls, audio, sr, params=None):
        """Apply all filters in sequence"""
        # Apply filters in sequence with parameters from Setup
        audio = cls.spectral_gating(audio, sr)
        audio = cls.wiener_filter(audio)
        audio = cls.gaussian_blur(audio)
        audio = cls.noise_gate(audio)
        audio = cls.dynamic_expansion(audio)
        audio = cls.exponential_smooth(audio)
        
        return audio