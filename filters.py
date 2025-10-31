import numpy as np
import librosa
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter1d

class AudioFilters:
    @staticmethod
    def noise_gate(data, threshold=0.01):
        """Apply noise gate filter"""
        return np.where(np.abs(data) > threshold, data, 0)

    @staticmethod
    def dynamic_expansion(data, threshold=0.35, ratio=1.5):
        """Apply dynamic expansion"""
        expanded = np.where(
            np.abs(data) > threshold,
            np.sign(data) * (np.abs(data) ** ratio),
            data
        )
        return expanded / np.max(np.abs(expanded))

    @staticmethod
    def exponential_smooth(data, alpha=0.9):
        """Apply exponential smoothing"""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for t in range(1, len(data)):
            smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
        return smoothed

    @staticmethod
    def spectral_gating(noisy_signal, sr):
        """Apply spectral gating for noise reduction"""
        stft = librosa.stft(noisy_signal, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        noise_thresh = np.median(magnitude, axis=1)[:, None]
        mask = magnitude > (1.5 * noise_thresh)
        
        filtered_stft = stft * mask
        return librosa.istft(filtered_stft, hop_length=512)

    @staticmethod
    def wiener_filter(audio, mysize=15, noise_var=0.01):
        """Apply Wiener filter"""
        return wiener(audio, mysize=mysize, noise=noise_var)

    @staticmethod
    def gaussian_blur(audio, sigma=2):
        """Apply Gaussian blur"""
        return gaussian_filter1d(audio, sigma=sigma)

    @classmethod
    def apply_all_filters(cls, audio, sr, params=None):
        """Apply all filters in sequence"""
        if params is None:
            params = {'wiener_size': 15, 'gaussian_sigma': 2}
        
        # Apply filters in sequence
        audio = cls.spectral_gating(audio, sr)
        audio = cls.wiener_filter(audio, mysize=params.get('wiener_size', 15))
        audio = cls.gaussian_blur(audio, sigma=params.get('gaussian_sigma', 2))
        audio = cls.noise_gate(audio)
        audio = cls.dynamic_expansion(audio)
        audio = cls.exponential_smooth(audio)
        
        return audio