import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import pygame
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
        pygame.mixer.init()
        self.is_playing = False
        self.current_player = None  # 'original' or 'processed'
        self.play_thread = None
        self.audio_length = 0  # Length of audio in seconds
        self.update_time_thread = None
        self.is_updating_slider = False
        
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
        
        # Time slider for original audio
        self.original_slider = ttk.Scale(original_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                       command=lambda x: self._seek_audio('original'))
        self.original_slider.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.original_time_label = ttk.Label(original_frame, text="0:00 / 0:00")
        self.original_time_label.grid(row=2, column=0, columnspan=3)

        # Processed audio frame (initially hidden)
        self.processed_frame = ttk.Frame(self.playback_frame)
        self.processed_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        self.processed_frame.grid_remove()  # Hide initially
        
        ttk.Label(self.processed_frame, text="Processed Audio:").grid(row=0, column=0, padx=5)
        ttk.Button(self.processed_frame, text="Play", command=lambda: self._play_audio('processed')).grid(row=0, column=1, padx=2)
        ttk.Button(self.processed_frame, text="Stop", command=self._stop_audio).grid(row=0, column=2, padx=2)
        
        # Time slider for processed audio
        self.processed_slider = ttk.Scale(self.processed_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                        command=lambda x: self._seek_audio('processed'))
        self.processed_slider.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.processed_time_label = ttk.Label(self.processed_frame, text="0:00 / 0:00")
        self.processed_time_label.grid(row=2, column=0, columnspan=3)

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
            
            # Get audio length
            audio_data = self.audio_processor.get_audio(self.current_audio_path)
            self.audio_length = len(audio_data) / 16000  # Sample rate is 16000
            
            # Update original audio time label
            self.original_time_label.config(text=f"0:00 / {int(self.audio_length//60)}:{int(self.audio_length%60):02d}")
            self.original_slider.set(0)

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
            
            # Update processed audio time label
            self.processed_time_label.config(text=f"0:00 / {int(self.audio_length//60)}:{int(self.audio_length%60):02d}")
            self.processed_slider.set(0)
            
            if self.show_spectrograms.get():
                self._create_spectrogram_window()

            # Save processed audio
            output_path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")],
                initialfile="processed_audio.wav"
            )
            
            if output_path:
                self.audio_processor.save_audio(predicted_audio, output_path)
                self.status_var.set("Processing completed successfully")
                messagebox.showinfo("Success", "Audio processing completed")
            
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
            pygame.mixer.music.load(self.current_audio_path)
            current_slider = self.original_slider
        else:  # processed
            # Save temporary file for processed audio
            temp_path = "temp_processed.wav"
            self.audio_processor.save_audio(self.processed_audio, temp_path)
            pygame.mixer.music.load(temp_path)
            current_slider = self.processed_slider
            
        self.is_playing = True
        self.current_player = audio_type
        
        # Start playing from slider position
        start_pos = current_slider.get() / 100.0
        pygame.mixer.music.play(start=start_pos * self.audio_length)
        
        # Start monitoring thread
        self.play_thread = threading.Thread(target=self._monitor_playback)
        self.play_thread.daemon = True
        self.play_thread.start()
        
        # Start time update thread
        self.update_time_thread = threading.Thread(target=self._update_time)
        self.update_time_thread.daemon = True
        self.update_time_thread.start()

    def _stop_audio(self):
        """Stop audio playback"""
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.current_player = None

    def _monitor_playback(self):
        """Monitor audio playback and update status"""
        while self.is_playing and pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        self.is_playing = False
        self.current_player = None
        
        # Reset slider position when playback ends
        if self.current_player == 'original':
            self.original_slider.set(0)
        else:
            self.processed_slider.set(0)

    def _update_time(self):
        """Update time labels and sliders during playback"""
        while self.is_playing and pygame.mixer.music.get_busy():
            if not self.is_updating_slider:
                current_pos = pygame.mixer.music.get_pos() / 1000.0  # Current position in seconds
                
                # Update appropriate slider and label
                if self.current_player == 'original':
                    self.original_slider.set((current_pos / self.audio_length) * 100)
                    mins, secs = divmod(int(current_pos), 60)
                    total_mins, total_secs = divmod(int(self.audio_length), 60)
                    self.original_time_label.config(
                        text=f"{mins}:{secs:02d} / {total_mins}:{total_secs:02d}")
                else:
                    self.processed_slider.set((current_pos / self.audio_length) * 100)
                    mins, secs = divmod(int(current_pos), 60)
                    total_mins, total_secs = divmod(int(self.audio_length), 60)
                    self.processed_time_label.config(
                        text=f"{mins}:{secs:02d} / {total_mins}:{total_secs:02d}")
            
            time.sleep(0.1)

    def _seek_audio(self, audio_type):
        """Handle slider position change"""
        if not self.is_playing:
            return
            
        self.is_updating_slider = True
        if audio_type == 'original':
            pos = self.original_slider.get()
        else:
            pos = self.processed_slider.get()
            
        # Calculate position in seconds
        seek_time = (pos / 100.0) * self.audio_length
        
        # Stop and restart playback at new position
        pygame.mixer.music.stop()
        if audio_type == 'original':
            pygame.mixer.music.load(self.current_audio_path)
        else:
            temp_path = "temp_processed.wav"
            pygame.mixer.music.load(temp_path)
            
        pygame.mixer.music.play(start=seek_time)
        self.is_updating_slider = False

    def __del__(self):
        """Cleanup when the application closes"""
        pygame.mixer.quit()
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