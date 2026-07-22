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
