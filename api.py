import webview
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import json
import os
import time

from audio_processor import AudioProcessor
from model_handler import ModelHandler
from filters import AudioFilters

SETTINGS_FILE = "settings.json"


class Api:
    def __init__(self):
        self.window = None
        self.audio_processor = AudioProcessor()
        self.model_handler = None
        self.current_audio_path = None
        self.processed_audio = None

        # Playback state
        self.is_playing = False
        self.current_player = None
        self.stream = None
        self.current_frame = 0
        self.audio_data = None
        self.sample_rate = 16000
        self.total_duration = 0
        self._update_thread = None

        # Processing state
        self._progress = 0.0
        self._status = "idle"

    def set_window(self, window):
        self.window = window

    # ── Model ────────────────────────────────────────────────────────────────

    def load_model(self):
        """Called from JS after window is ready."""
        try:
            self.model_handler = ModelHandler("model/nocle.hdf5", self.audio_processor)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── File ─────────────────────────────────────────────────────────────────

    def browse_file(self):
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Audio Files (*.wav;*.mp3;*.flac;*.ogg;*.aac;*.m4a)", "All Files (*.*)")
        )
        if result:
            path = result[0]
            return self.get_audio_info(path)
        return {"success": False, "cancelled": True}

    def get_audio_info(self, path):
        try:
            self.current_audio_path = path
            audio = self.audio_processor.get_audio(path)
            duration = len(audio) / self.audio_processor.target_sample_rate
            return {
                "success": True,
                "path": path,
                "filename": os.path.basename(path),
                "duration": round(duration, 2),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Processing ────────────────────────────────────────────────────────────

    def process_audio(self, options):
        """Start audio processing in a background thread."""
        if not self.current_audio_path:
            return {"success": False, "error": "No file selected"}
        if not self.model_handler:
            return {"success": False, "error": "Model not loaded"}

        def task():
            try:
                self._progress = 0.05
                self._status = "processing"

                predicted = self.model_handler.predict(self.current_audio_path)
                self._progress = 0.6

                use_any_filter = options.get("spectral_gate") or options.get("wiener") or options.get("gaussian")
                if use_any_filter:
                    params = {
                        "wiener_size": int(options.get("wiener_size", 15)),
                        "gaussian_sigma": float(options.get("gaussian_sigma", 2.0)),
                    }
                    predicted = AudioFilters.apply_all_filters(predicted, sr=16000, params=params)
                self._progress = 0.85

                if options.get("normalize", True):
                    predicted = self.audio_processor.normalize_audio(predicted)
                self._progress = 1.0

                self.processed_audio = predicted
                duration = round(len(predicted) / 16000, 2)
                self._status = "done"
                self.window.evaluate_js(f'App.onProcessingDone({{duration: {duration}}})')

            except Exception as e:
                self._status = "error"
                safe = str(e).replace('"', '\\"').replace("'", "\\'")
                self.window.evaluate_js(f'App.onProcessingError("{safe}")')

        threading.Thread(target=task, daemon=True).start()
        return {"success": True}

    def get_progress(self):
        return {"progress": self._progress, "status": self._status}

    # ── Playback ─────────────────────────────────────────────────────────────

    def play_audio(self, audio_type):
        if audio_type == "original" and not self.current_audio_path:
            return {"success": False, "error": "No file loaded"}
        if audio_type == "processed" and self.processed_audio is None:
            return {"success": False, "error": "No processed audio"}

        self.stop_audio()
        time.sleep(0.1)

        if audio_type == "original":
            data, sr = sf.read(self.current_audio_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
        else:
            data = self.processed_audio
            sr = 16000

        self.audio_data = data.astype(np.float32)
        self.sample_rate = sr
        self.total_duration = len(data) / sr
        self.current_frame = 0
        self.is_playing = True
        self.current_player = audio_type

        def callback(outdata, frames, t, status):
            if not self.is_playing:
                raise sd.CallbackStop()
            chunk = self.audio_data[self.current_frame: self.current_frame + frames]
            if len(chunk) < frames:
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):] = 0
                raise sd.CallbackStop()
            outdata[:, 0] = chunk
            self.current_frame += frames

        self.stream = sd.OutputStream(
            samplerate=sr, channels=1, callback=callback,
            finished_callback=self._on_finished
        )
        self.stream.start()

        self._update_thread = threading.Thread(target=self._push_time, daemon=True)
        self._update_thread.start()
        return {"success": True, "duration": self.total_duration}

    def stop_audio(self):
        self.is_playing = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.current_player = None
        return {"success": True}

    def _on_finished(self):
        self.is_playing = False
        self.current_player = None
        try:
            self.window.evaluate_js("App.onPlaybackFinished()")
        except Exception:
            pass

    def _push_time(self):
        last = -1
        while self.is_playing:
            pos = self.current_frame / max(self.sample_rate, 1)
            pos_int = int(pos)
            if pos_int != last:
                last = pos_int
                try:
                    self.window.evaluate_js(
                        f'App.onTimeUpdate({pos:.2f}, {self.total_duration:.2f}, "{self.current_player}")'
                    )
                except Exception:
                    break
            time.sleep(0.1)

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_audio(self):
        if self.processed_audio is None:
            return {"success": False, "error": "Nothing to save"}
        result = self.window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename="processed_audio.wav",
            file_types=("WAV Audio (*.wav)",)
        )
        if result:
            path = result if isinstance(result, str) else result[0]
            if not path.lower().endswith(".wav"):
                path += ".wav"
            try:
                self.audio_processor.save_audio(self.processed_audio, path)
                return {"success": True, "path": path}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "cancelled": True}

    # ── Real-Time Mic Stream ──────────────────────────────────────────────

    def get_audio_devices(self):
        try:
            devices = sd.query_devices()
            default_in = sd.default.device[0]
            default_out = sd.default.device[1]

            inputs = []
            outputs = []
            for idx, d in enumerate(devices):
                # Filter out Sound Mapper wrappers for primary selection if desired, or include them
                name = d['name']
                is_default_in = (idx == default_in)
                is_default_out = (idx == default_out)

                if d['max_input_channels'] > 0:
                    inputs.append({'index': idx, 'name': name, 'is_default': is_default_in})
                if d['max_output_channels'] > 0:
                    outputs.append({'index': idx, 'name': name, 'is_default': is_default_out})

            # If default_in is Sound Mapper, try to pick the first hardware microphone
            selected_input = default_in
            if default_in >= 0 and default_in < len(devices) and 'Sound Mapper' in devices[default_in]['name']:
                for item in inputs:
                    if 'Sound Mapper' not in item['name'] and 'Primary' not in item['name']:
                        selected_input = item['index']
                        break

            return {
                'success': True,
                'inputs': inputs,
                'outputs': outputs,
                'default_input': selected_input,
                'default_output': default_out
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def set_windows_default_mic(self, device_name="CABLE Output"):
        """Set specified device as Windows default recording device using NirCmd."""
        try:
            nircmd_path = "nircmd.exe"
            if not os.path.exists(nircmd_path):
                return {'success': False, 'error': 'nircmd.exe not found'}

            subprocess.run([nircmd_path, "setdefaultsounddevice", device_name, "1"], capture_output=True)
            subprocess.run([nircmd_path, "setdefaultsounddevice", device_name, "2"], capture_output=True)
            return {'success': True, 'device': device_name}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def start_realtime_mic(self, input_idx, output_idx, buffer_size=4096):
        if hasattr(self, '_rt_running') and self._rt_running:
            self.stop_realtime_mic()

        self._rt_running = True
        self._rt_buffer_size = int(buffer_size)

        def rt_thread():
            try:
                def callback(indata, outdata, frames, time_info, status):
                    if not self._rt_running:
                        raise sd.CallbackStop()

                    audio_chunk = indata[:, 0]
                    # Apply noise reduction model / filters if loaded
                    if self.model_handler:
                        try:
                            # Preprocess chunk to target batch size expected by model if needed
                            if len(audio_chunk) == 12000:
                                processed = self.model_handler.predict_batch(audio_chunk)
                            else:
                                processed = audio_chunk # Fallback to pass-through if mismatch
                        except Exception:
                            processed = audio_chunk
                    else:
                        processed = audio_chunk

                    outdata[:, 0] = processed

                self._rt_stream = sd.Stream(
                    device=(int(input_idx), int(output_idx)),
                    samplerate=16000,
                    blocksize=self._rt_buffer_size,
                    channels=1,
                    callback=callback
                )
                self._rt_stream.start()
                while self._rt_running:
                    time.sleep(0.1)
            except Exception as e:
                self._rt_running = False
                safe = str(e).replace('"', '\\"').replace("'", "\\'")
                try:
                    self.window.evaluate_js(f'App.onRealtimeError("{safe}")')
                except Exception:
                    pass

        threading.Thread(target=rt_thread, daemon=True).start()
        return {'success': True}

    def stop_realtime_mic(self):
        self._rt_running = False
        if hasattr(self, '_rt_stream') and self._rt_stream:
            try:
                self._rt_stream.stop()
                self._rt_stream.close()
            except Exception:
                pass
            self._rt_stream = None
        return {'success': True}

    # ── Settings ─────────────────────────────────────────────────────────────

    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save_settings(self, data):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

