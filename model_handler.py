import tensorflow as tf
import numpy as np

class ModelHandler:
    def __init__(self, model_path, audio_processor):
        from setup import Setup
        self.model = tf.keras.models.load_model(model_path)
        # Compile the model with configured optimizer and loss
        self.model.compile(optimizer=Setup.MODEL_OPTIMIZER, loss=Setup.MODEL_LOSS)
        self.audio_processor = audio_processor

    def predict(self, path, batching_size=12000, use_filters=False, filter_params=None):
        """Make prediction using the model"""
        audio_batches = self.audio_processor.get_audio_in_batches(path, batching_size)
        original_length = len(self.audio_processor.get_audio(path))
        
        predicted_batches = []
        for batch in audio_batches:
            frame = tf.squeeze(
                self.model.predict(
                    tf.expand_dims(tf.expand_dims(batch, -1), 0),
                    verbose=0
                )
            )
            predicted_batches.append(frame.numpy())
        
        predicted_audio = np.concatenate(predicted_batches)
        predicted_audio = predicted_audio[:original_length]
        
        return predicted_audio

    def predict_tflite(self, path, tflite_model_path, batching_size=12000):
        """Make prediction using TFLite model"""
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        audio_batches = self.audio_processor.get_audio_in_batches(path, batching_size)
        original_length = len(self.audio_processor.get_audio(path))
        
        predicted_batches = []
        for batch in audio_batches:
            input_data = np.expand_dims(np.expand_dims(batch, -1), 0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            frame = interpreter.get_tensor(output_details[0]['index'])
            predicted_batches.append(frame.squeeze())
        
        predicted_audio = np.concatenate(predicted_batches)
        return predicted_audio[:original_length]