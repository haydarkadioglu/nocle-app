import tensorflow as tf
import numpy as np
import librosa

from setup import Setup

class AudioProcessor:
    def __init__(self, target_sample_rate=Setup.SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate

    def get_audio_in_batches(self, path, batching_size=12000):
        """Load and process audio file in batches"""
        audio, sample_rate = tf.audio.decode_wav(
            tf.io.read_file(path), desired_channels=1)
        audio_np = audio.numpy().squeeze()

        if sample_rate != self.target_sample_rate:
            audio_np = librosa.resample(
                audio_np, 
                orig_sr=sample_rate.numpy(), 
                target_sr=self.target_sample_rate
            )

        audio_batches = []
        total_samples = len(audio_np)
        
        for i in range(0, total_samples, batching_size):
            batch = audio_np[i:i + batching_size]
            if len(batch) < batching_size:
                batch = np.pad(batch, (0, batching_size - len(batch)), mode='constant')
            audio_batches.append(batch)

        return tf.stack(audio_batches)

    def get_audio(self, path):
        """Load complete audio file"""
        audio, sample_rate = tf.audio.decode_wav(
            tf.io.read_file(path), desired_channels=1)
        audio_np = audio.numpy().squeeze()

        if sample_rate != self.target_sample_rate:
            audio_np = librosa.resample(
                audio_np, 
                orig_sr=sample_rate.numpy(), 
                target_sr=self.target_sample_rate
            )

        return audio_np

    def save_audio(self, audio_data, output_path):
        """Save audio data to file"""
        audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        audio_tensor = tf.reshape(audio_tensor, [-1, 1])
        tf.io.write_file(
            output_path,
            tf.audio.encode_wav(audio_tensor, sample_rate=self.target_sample_rate)
        )