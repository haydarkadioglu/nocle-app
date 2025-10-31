import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import sounddevice as sd
import soundfile as sf
import threading
import time
import librosa.display
from audio_processor import AudioProcessor
from model_handler import ModelHandler
from filters import AudioFilters

class NocleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nocle Audio Processing")
        self.root.geometry("800x700")
        
        # Initialize components
        self.audio_processor = AudioProcessor()
        self.model_handler = None
        self.current_audio_path = None
        self.output_path = None
        self.processed_audio = None
        self.spectrogram_window = None
        self.fig_original = None
        self.fig_processed = None
        self.ax_original = None
        self.ax_processed = None
        self.canvas_original = None
        self.canvas_processed = None
        
        # Audio playback components
        self.is_playing = False
        self.current_player = None  # 'original' or 'processed'
        self.play_thread = None
        self.update_time_thread = None
        self.stream = None
        self.current_frame = 0
        self.audio_data = None
        self.sample_rate = 16000
        
        self._create_widgets()
        self._load_model()

    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection
        ttk.Label(main_frame, text="Audio File:").grid(row=0, column=0, sticky=tk.W)
        self.file_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self._browse_file).grid(row=0, column=2)

        # Filter options
        filter_frame = ttk.LabelFrame(main_frame, text="Filter Options", padding="5")
        filter_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        # Checkboxes for filters
        self.use_spectral_gate = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Spectral Gate", variable=self.use_spectral_gate).grid(row=0, column=0)

        self.use_wiener = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Wiener Filter", variable=self.use_wiener).grid(row=0, column=1)

        self.use_gaussian = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Gaussian Blur", variable=self.use_gaussian).grid(row=0, column=2)

        # Show Spectrograms option
        self.show_spectrograms = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Show Spectrograms", 
                       variable=self.show_spectrograms).grid(row=0, column=3)

        # Filter parameters
        param_frame = ttk.Frame(filter_frame)
        param_frame.grid(row=1, column=0, columnspan=3, pady=5)

        ttk.Label(param_frame, text="Wiener Size:").grid(row=0, column=0)
        self.wiener_size = ttk.Spinbox(param_frame, from_=3, to=31, increment=2, width=5)
        self.wiener_size.set(15)
        self.wiener_size.grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="Gaussian Sigma:").grid(row=0, column=2, padx=5)
        self.gaussian_sigma = ttk.Spinbox(param_frame, from_=0.1, to=5.0, increment=0.1, width=5)
        self.gaussian_sigma.set(2.0)
        self.gaussian_sigma.grid(row=0, column=3, padx=5)

        # Process button
        ttk.Button(main_frame, text="Process Audio", command=self._process_audio).grid(row=2, column=0, columnspan=3, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=4, column=0, columnspan=3)

        # Audio playback frame (initially hidden)
        self.playback_frame = ttk.LabelFrame(main_frame, text="Audio Controls", padding="5")
        self.playback_frame.grid(row=5, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        self.playback_frame.grid_remove()  # Hide initially

        # Original audio controls
        original_frame = ttk.Frame(self.playback_frame)
        original_frame.grid(row=0, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(original_frame, text="Original Audio:").grid(row=0, column=0, padx=5)
        ttk.Button(original_frame, text="Play", command=lambda: self._play_audio('original')).grid(row=0, column=1, padx=2)
        ttk.Button(original_frame, text="Stop", command=self._stop_audio).grid(row=0, column=2, padx=2)
        
        # Time display for original audio
        ttk.Label(original_frame, text="Time (s):").grid(row=1, column=0, padx=5)
        self.original_time_label = ttk.Label(original_frame, text="0")
        self.original_time_label.grid(row=1, column=1)

        # Processed audio frame (initially hidden)
        self.processed_frame = ttk.Frame(self.playback_frame)
        self.processed_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        self.processed_frame.grid_remove()  # Hide initially
        
        ttk.Label(self.processed_frame, text="Processed Audio:").grid(row=0, column=0, padx=5)
        ttk.Button(self.processed_frame, text="Play", command=lambda: self._play_audio('processed')).grid(row=0, column=1, padx=2)
        ttk.Button(self.processed_frame, text="Stop", command=self._stop_audio).grid(row=0, column=2, padx=2)
        
        # Time display for processed audio
        ttk.Label(self.processed_frame, text="Time (s):").grid(row=1, column=0, padx=5)
        self.processed_time_label = ttk.Label(self.processed_frame, text="0")
        self.processed_time_label.grid(row=1, column=1)

    def _load_model(self):
        try:
            model_path = "model/nocle.hdf5"
            self.model_handler = ModelHandler(model_path, self.audio_processor)
            self.status_var.set("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.quit()

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.current_audio_path = file_path
            
            # Show original audio controls
            self.playback_frame.grid()
            
            # Calculate total duration and update display
            audio_data = self.audio_processor.get_audio(self.current_audio_path)
            total_duration = int(len(audio_data) / 16000)  # Sample rate is 16000
            self.original_time_label.config(text=f"0 / {total_duration}")

    def _process_audio(self):
        if not self.current_audio_path:
            messagebox.showwarning("Warning", "Please select an audio file first")
            return

        try:
            self.status_var.set("Processing audio...")
            self.progress_var.set(20)
            self.root.update()

            # Get filter parameters
            filter_params = {
                'wiener_size': int(self.wiener_size.get()),
                'gaussian_sigma': float(self.gaussian_sigma.get())
            }

            # Process audio
            predicted_audio = self.model_handler.predict(self.current_audio_path)
            self.progress_var.set(60)
            self.root.update()

            # Apply selected filters
            if any([self.use_spectral_gate.get(), self.use_wiener.get(), self.use_gaussian.get()]):
                predicted_audio = AudioFilters.apply_all_filters(
                    predicted_audio,
                    sr=16000,
                    params=filter_params
                )

            self.progress_var.set(80)
            self.root.update()

            # Store processed audio and update UI
            self.processed_audio = predicted_audio
            
            # Show processed audio controls
            self.processed_frame.grid()
            
            # Update processed audio time label with total duration
            total_duration = int(len(predicted_audio) / 16000)
            self.processed_time_label.config(text=f"0 / {total_duration}")
            
            if self.show_spectrograms.get():
                self._create_spectrogram_window()

            # Show save button for processed audio
            self.status_var.set("Processing completed successfully")
            
            # Add save button to processed frame if not already added
            if not hasattr(self, 'save_button'):
                self.save_button = ttk.Button(self.processed_frame, text="Save", command=self._save_processed_audio)
                self.save_button.grid(row=0, column=3, padx=2)
            
            self.progress_var.set(100)

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed")
        
        finally:
            self.progress_var.set(0)

    def _plot_spectrogram(self, audio_data, ax, sr=16000):
        """Plot spectrogram on the given axes"""
        ax.clear()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Spectrogram')
        if ax == self.ax_original:
            self.fig_original.colorbar(img, ax=ax, format="%+2.f dB")
        else:
            self.fig_processed.colorbar(img, ax=ax, format="%+2.f dB")

    def _create_spectrogram_window(self):
        """Create a new window for spectrograms"""
        if self.spectrogram_window is None or not self.spectrogram_window.winfo_exists():
            self.spectrogram_window = tk.Toplevel(self.root)
            self.spectrogram_window.title("Audio Spectrograms")
            self.spectrogram_window.geometry("1000x700")

            # Create figures for spectrograms
            self.fig_original = Figure(figsize=(8, 4))
            self.ax_original = self.fig_original.add_subplot(111)
            self.canvas_original = FigureCanvasTkAgg(self.fig_original, master=self.spectrogram_window)
            self.canvas_original.get_tk_widget().pack(pady=10)
            ttk.Label(self.spectrogram_window, text="Original Audio Spectrogram").pack()

            self.fig_processed = Figure(figsize=(8, 4))
            self.ax_processed = self.fig_processed.add_subplot(111)
            self.canvas_processed = FigureCanvasTkAgg(self.fig_processed, master=self.spectrogram_window)
            self.canvas_processed.get_tk_widget().pack(pady=10)
            ttk.Label(self.spectrogram_window, text="Processed Audio Spectrogram").pack()

            # Update original spectrogram if file is loaded
            if self.current_audio_path:
                self._update_original_spectrogram()

            # Update processed spectrogram if available
            if self.processed_audio is not None:
                self._update_processed_spectrogram()

    def _toggle_spectrograms(self):
        """Handle spectrogram visibility"""
        if self.show_spectrograms.get():
            self._create_spectrogram_window()
        else:
            if self.spectrogram_window and self.spectrogram_window.winfo_exists():
                self.spectrogram_window.destroy()
                self.spectrogram_window = None

    def _update_original_spectrogram(self):
        """Update the original audio spectrogram"""
        if self.current_audio_path and self.show_spectrograms.get():
            if self.spectrogram_window is None or not self.spectrogram_window.winfo_exists():
                self._create_spectrogram_window()
            audio_data = self.audio_processor.get_audio(self.current_audio_path)
            self._plot_spectrogram(audio_data, self.ax_original)
            self.canvas_original.draw()

    def _update_processed_spectrogram(self):
        """Update the processed audio spectrogram"""
        if self.processed_audio is not None and self.show_spectrograms.get():
            if self.spectrogram_window is None or not self.spectrogram_window.winfo_exists():
                self._create_spectrogram_window()
            self._plot_spectrogram(self.processed_audio, self.ax_processed)
            self.canvas_processed.draw()

    def _play_audio(self, audio_type):
        """Play either original or processed audio"""
        if audio_type == 'original' and not self.current_audio_path:
            messagebox.showwarning("Warning", "Please select an audio file first")
            return
        if audio_type == 'processed' and self.processed_audio is None:
            messagebox.showwarning("Warning", "Please process an audio file first")
            return

        self._stop_audio()  # Stop any currently playing audio
        
        if audio_type == 'original':
            data, sr = sf.read(self.current_audio_path)
            self.current_time_label = self.original_time_label
        else:  # processed
            data = self.processed_audio
            sr = 16000
            self.current_time_label = self.processed_time_label
            
        # Calculate and store total duration
        self.total_duration = int(len(data) / sr)
        
        self.audio_data = data
        self.sample_rate = sr
        self.current_frame = 0  # Start from beginning
        
        def callback(outdata, frames, time, status):
            if status:
                print(status)
            if not self.is_playing:
                raise sd.CallbackStop()
            
            chunk = self.audio_data[self.current_frame:self.current_frame + frames]
            if len(chunk) < frames:
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = chunk
                self.current_frame += frames
        
        self.is_playing = True
        self.current_player = audio_type
        
        # Start audio stream
        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            callback=callback,
            finished_callback=self._on_playback_finished
        )
        self.stream.start()
        
        # Stop previous update thread if exists
        if hasattr(self, 'update_time_thread') and self.update_time_thread:
            self.is_playing = False
            self.update_time_thread.join(timeout=1.0)
        
        # Start new time update thread
        self.is_playing = True
        self.update_time_thread = threading.Thread(target=self._update_time)
        self.update_time_thread.daemon = True
        self.update_time_thread.start()

    def _stop_audio(self):
        """Stop audio playback"""
        # Stop playback first
        self.is_playing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for update thread to finish
        if hasattr(self, 'update_time_thread') and self.update_time_thread:
            self.update_time_thread.join(timeout=1.0)
            self.update_time_thread = None
        
        self.current_player = None
        
        # Reset time display
        if hasattr(self, 'current_time_label'):
            self.root.after_idle(
                lambda: self.current_time_label.config(text=f"0 / {self.total_duration}")
            )

    def _on_playback_finished(self):
        """Called when playback is finished"""
        self.is_playing = False
        self.current_player = None
        self.stream = None
        self.root.after(0, self._reset_slider)

    def _reset_slider(self):
        """Reset slider position when playback ends"""
        if self.current_player == 'original':
            self.original_slider.set(0)
        else:
            self.processed_slider.set(0)

    def _update_time(self):
        """Update time display during playback"""
        last_pos = -1  # Track last position to avoid unnecessary updates
        
        while self.is_playing:
            try:
                if self.audio_data is not None and hasattr(self, 'current_frame'):
                    current_pos = int(self.current_frame / self.sample_rate)
                    
                    # Only update if position changed
                    if current_pos != last_pos:
                        last_pos = current_pos
                        # Update time label in the main thread
                        self.root.after(10, lambda p=current_pos: 
                            self.current_time_label.config(text=f"{p} / {self.total_duration}")
                        )
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Error updating time: {e}")
                break

    def _on_playback_finished(self):
        """Called when playback is finished"""
        self.is_playing = False
        self.current_player = None
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Reset time display
        self.root.after_idle(
            lambda: self.current_time_label.config(text="0")
        )

    def _save_processed_audio(self):
        """Save the processed audio to a user-selected location"""
        if self.processed_audio is None:
            messagebox.showwarning("Warning", "No processed audio to save")
            return
            
        output_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile="processed_audio.wav"
        )
        
        if output_path:
            self.audio_processor.save_audio(self.processed_audio, output_path)
            messagebox.showinfo("Success", "Audio saved successfully")

    def __del__(self):
        """Cleanup when the application closes"""
        self._stop_audio()  # Stop any playing audio
        # Remove temporary file if it exists
        if os.path.exists("temp_processed.wav"):
            try:
                os.remove("temp_processed.wav")
            except:
                pass

def main():
    root = tk.Tk()
    app = NocleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()