import flet as ft
import os
import numpy as np
import matplotlib
matplotlib.use('svg')
from matplotlib.figure import Figure
import librosa
import sounddevice as sd
import soundfile as sf
import threading
import time
import librosa.display
from audio_processor import AudioProcessor
from model_handler import ModelHandler
from filters import AudioFilters
import io
import base64

class NocleGUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Nocle Audio Processing"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20
        self.page.scroll = ft.ScrollMode.AUTO
        
        # Flet Snackbar for messages
        self.page.snack_bar = ft.SnackBar(ft.Text(""), open=False)

        # Initialize components
        self.audio_processor = AudioProcessor()
        self.model_handler = None
        self.current_audio_path = None
        self.output_path = None
        self.processed_audio = None
        
        # Audio playback components
        self.is_playing = False
        self.current_player = None
        self.play_thread = None
        self.update_time_thread = None
        self.stream = None
        self.current_frame = 0
        self.audio_data = None
        self.sample_rate = 16000
        self.total_duration = 0
        
        self.setup_ui()
        self._load_model()

    def show_message(self, text, is_error=False):
        self.page.snack_bar.content = ft.Text(text)
        self.page.snack_bar.bgcolor = ft.colors.ERROR if is_error else ft.colors.GREEN_700
        self.page.snack_bar.open = True
        self.page.update()

    def setup_ui(self):
        # File Picker setup
        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.save_file_picker = ft.FilePicker(on_result=self.on_file_saved)
        self.page.overlay.extend([self.file_picker, self.save_file_picker])

        # Header
        header = ft.Row([
            ft.Icon(ft.icons.MULTITRACK_AUDIO_ROUNDED, size=40, color=ft.colors.AMBER_400),
            ft.Text("Nocle Audio Enhancer", size=32, weight=ft.FontWeight.BOLD, color=ft.colors.INDIGO_300)
        ], alignment=ft.MainAxisAlignment.CENTER)

        # File Selection Area
        self.file_path_text = ft.TextField(label="Selected Audio File", read_only=True, expand=True)
        browse_btn = ft.ElevatedButton(
            "Browse File", 
            icon=ft.icons.FOLDER_OPEN,
            on_click=lambda _: self.file_picker.pick_files(allowed_extensions=["wav"], allow_multiple=False),
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
        )
        file_row = ft.Row([self.file_path_text, browse_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        # Filter Options Panel
        self.use_spectral_gate = ft.Switch(label="Spectral Gate", value=False)
        self.use_wiener = ft.Switch(label="Wiener Filter", value=False)
        self.use_gaussian = ft.Switch(label="Gaussian Blur", value=False)
        self.show_spectrograms = ft.Switch(label="Show Spectrograms", value=False)
        
        filter_switches = ft.Row([self.use_spectral_gate, self.use_wiener, self.use_gaussian, self.show_spectrograms], wrap=True)

        self.wiener_size = ft.TextField(label="Wiener Size", value="15", width=120, keyboard_type=ft.KeyboardType.NUMBER)
        self.gaussian_sigma = ft.TextField(label="Gaussian Sigma", value="2.0", width=120, keyboard_type=ft.KeyboardType.NUMBER)
        filter_params = ft.Row([self.wiener_size, self.gaussian_sigma])

        filter_panel = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Filter Settings", size=20, weight=ft.FontWeight.W_600),
                    ft.Divider(height=10),
                    filter_switches,
                    ft.Container(height=5),
                    filter_params
                ]),
                padding=20
            ),
            elevation=2
        )

        # Processing section
        self.process_btn = ft.ElevatedButton(
            "Process Audio", 
            icon=ft.icons.AUTO_AWESOME,
            on_click=self._process_audio,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                bgcolor=ft.colors.INDIGO_700,
                color=ft.colors.WHITE
            ),
            width=200, height=50
        )
        self.progress_bar = ft.ProgressBar(value=0, visible=False, expand=True)
        self.status_text = ft.Text("Ready", italic=True)
        
        process_row = ft.Row([self.process_btn, self.progress_bar], alignment=ft.MainAxisAlignment.START)
        
        # Audio Players section
        # Original
        self.original_time = ft.Text("0:00 / 0:00", weight=ft.FontWeight.BOLD)
        self.original_play_btn = ft.IconButton(icon=ft.icons.PLAY_ARROW, on_click=lambda _: self._play_audio('original'), disabled=True, icon_color=ft.colors.GREEN_400)
        self.original_stop_btn = ft.IconButton(icon=ft.icons.STOP, on_click=lambda _: self._stop_audio(), disabled=True, icon_color=ft.colors.RED_400)
        
        original_player = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Original Audio", weight=ft.FontWeight.BOLD, size=16),
                    ft.Divider(),
                    ft.Row([self.original_play_btn, self.original_stop_btn, self.original_time])
                ]),
                padding=15
            ),
            expand=True
        )
        
        # Processed
        self.processed_time = ft.Text("0:00 / 0:00", weight=ft.FontWeight.BOLD)
        self.processed_play_btn = ft.IconButton(icon=ft.icons.PLAY_ARROW, on_click=lambda _: self._play_audio('processed'), disabled=True, icon_color=ft.colors.GREEN_400)
        self.processed_stop_btn = ft.IconButton(icon=ft.icons.STOP, on_click=lambda _: self._stop_audio(), disabled=True, icon_color=ft.colors.RED_400)
        self.save_btn = ft.ElevatedButton("Save", icon=ft.icons.SAVE, on_click=lambda _: self.save_file_picker.save_file(allowed_extensions=["wav"], file_name="processed_audio.wav"), disabled=True, bgcolor=ft.colors.AMBER_700)

        processed_player = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Processed Audio", weight=ft.FontWeight.BOLD, size=16),
                    ft.Divider(),
                    ft.Row([self.processed_play_btn, self.processed_stop_btn, self.processed_time, ft.Container(expand=True), self.save_btn])
                ]),
                padding=15
            ),
            expand=True
        )

        self.players_row = ft.Row([original_player, processed_player], expand=True, visible=False)

        # Spectrogram area
        self.spectrogram_container = ft.Column(visible=False)

        # Assembling the Main Audio Tab
        main_tab_content = ft.Column([
            header,
            ft.Divider(height=30),
            file_row,
            ft.Container(height=10),
            filter_panel,
            ft.Container(height=20),
            process_row,
            ft.Container(height=5),
            self.status_text,
            ft.Container(height=20),
            self.players_row,
            ft.Divider(height=30),
            self.spectrogram_container
        ], scroll=ft.ScrollMode.AUTO)

        # "Nasıl Kullanılır" (Guide) Tab Content
        guide_md = """
# Nocle Uygulamasına Hoş Geldiniz

**Nocle**, yapay zeka (Deep Learning) destekli, ses dosyalarındaki arka plan gürültülerini temizleyen güçlü bir uygulamadır. 

### 1. Temel Kullanım

1. **Dosya Seçin:** "Browse File" düğmesine tıklayın ve `.wav` formatındaki gürültülü ses kaydınızı yükleyin.
2. **Dinleyin:** Yüklendikten sonra 'Original Audio' kısmından orijinal kaydınızı dinleyebilirsiniz.
3. **Filtreleri Ayarlayın:** Özel ihtiyaçlarınıza göre çeşitli ses filtrelerini açıp kapatabilirsiniz (Aşağıda detayları açıklanmıştır).
4. **İşlemi Başlatın:** "Process Audio" butonuna bastığınızda yapay zeka devreye girer ve kısa bir süre içinde sesinizi temizler.
5. **Sonucu Karşılaştırın & Kaydedin:** Temizlenen sesi 'Processed Audio' bölümünden dinleyebilir ve "Save" diyerek cihazınıza indirebilirsiniz.

---

### 2. Filtreler Ne İşe Yarar?

Mevcut yapay zeka modeli seste oldukça iyi bir temizleme yapar. Ancak çıkan sesin *daha da pürüzsüz* duymasını istiyorsanız şu ek algoritmaları aktif edebilirsiniz:

- **Spectral Gate:** Sesteki *statik gürültülerin* (ör. klima, fan, arka plan uğultusu) eşik değerini belirleyip sadece o frekansları susturur. Kısık sesleri arka plandan siler.
- **Wiener Filter:** Ses sinyallerini tarar ve matematiskel olarak gürültü ihtimali olan yüksek frekanslı parazitleri sönümler. *(Wiener Size parametresi arttıkça temizleme agresifleşir ancak seste boğukluk yapabilir)*.
- **Gaussian Blur:** Sesteki ani yırtılmaları veya çok sivri patlamaları yumuşatır. Sesi mikslemek için daha kadifemsi yapar *(Gaussian Sigma parametresi bu etkinin alanını belirler)*.

> **Spektrogramları Göster (Show Spectrograms):** Seçtiğinizde grafik tablosuyla sesin frekans yapısını "Gürültülü" ve "Temizlenmiş" halini yan yana görmenizi sağlar.

"""
        guide_tab_content = ft.Container(
            content=ft.Markdown(
                guide_md,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            ),
            padding=20
        )

        # Assemble Tabs
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="🎵 Ses Temizleme (Ana Sayfa)",
                    content=main_tab_content,
                ),
                ft.Tab(
                    text="📖 Nasıl Kullanılır?",
                    content=guide_tab_content,
                ),
            ],
            expand=True,
        )

        self.page.add(tabs)

    def format_time(self, seconds):
        if seconds is None:
            return "0:00"
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _load_model(self):
        try:
            model_path = "model/nocle.hdf5"
            self.model_handler = ModelHandler(model_path, self.audio_processor)
            self.status_text.value = "Model loaded successfully"
            self.status_text.color = ft.colors.GREEN_400
        except Exception as e:
            self.show_message(f"Failed to load model: {str(e)}", is_error=True)
            self.status_text.value = "Failed to load model"
            self.status_text.color = ft.colors.RED_400
        self.page.update()

    def on_file_selected(self, e):
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            self.current_audio_path = file_path
            self.file_path_text.value = file_path
            
            try:
                audio_data = self.audio_processor.get_audio(self.current_audio_path)
                total_dur = int(len(audio_data) / 16000)
                
                self.original_time.value = f"0:00 / {self.format_time(total_dur)}"
                self.original_play_btn.disabled = False
                self.original_stop_btn.disabled = False
                self.players_row.visible = True
                
                if self.show_spectrograms.value:
                    self._update_spectrograms()
                    
                self.status_text.value = "Ready to process"
                self.status_text.color = ft.colors.WHITE
            except Exception as ex:
                self.show_message(f"Could not load audio: {str(ex)}", is_error=True)
            
            self.page.update()

    def _process_audio(self, e):
        if not self.current_audio_path:
            self.show_message("Please select an audio file first", is_error=True)
            return

        def process_task():
            try:
                self.status_text.value = "Processing audio..."
                self.status_text.color = ft.colors.AMBER_400
                self.progress_bar.visible = True
                self.progress_bar.value = 0.2
                self.page.update()

                filter_params = {
                    'wiener_size': int(self.wiener_size.value),
                    'gaussian_sigma': float(self.gaussian_sigma.value)
                }

                # Predict
                predicted_audio = self.model_handler.predict(self.current_audio_path)
                self.progress_bar.value = 0.6
                self.page.update()

                # Filters
                if any([self.use_spectral_gate.value, self.use_wiener.value, self.use_gaussian.value]):
                    predicted_audio = AudioFilters.apply_all_filters(
                        predicted_audio,
                        sr=16000,
                        params=filter_params
                    )

                self.progress_bar.value = 0.9
                self.page.update()

                self.processed_audio = predicted_audio
                total_dur = int(len(predicted_audio) / 16000)
                self.processed_time.value = f"0:00 / {self.format_time(total_dur)}"
                
                self.processed_play_btn.disabled = False
                self.processed_stop_btn.disabled = False
                self.save_btn.disabled = False

                if self.show_spectrograms.value:
                    self._update_spectrograms()

                self.status_text.value = "Processing completed successfully"
                self.status_text.color = ft.colors.GREEN_400
                self.progress_bar.value = 1.0
                self.show_message("Audio processed successfully!")

            except Exception as ex:
                self.show_message(f"Processing failed: {str(ex)}", is_error=True)
                self.status_text.value = "Processing failed"
                self.status_text.color = ft.colors.RED_400
                
            finally:
                self.progress_bar.visible = False
                self.page.update()

        # Run process logic in a separate thread so it doesn't freeze UI
        threading.Thread(target=process_task, daemon=True).start()

    def _update_spectrograms(self):
        self.spectrogram_container.controls.clear()
        
        if self.current_audio_path:
            audio_data = self.audio_processor.get_audio(self.current_audio_path)
            b64_orig = self._create_spectrogram_base64(audio_data, "Original Audio Spectrogram")
            self.spectrogram_container.controls.append(ft.Container(
                 ft.Text("Original Audio Spectrogram", weight="bold"), padding=5))
            self.spectrogram_container.controls.append(ft.Image(src_base64=b64_orig, expand=True, fit=ft.ImageFit.CONTAIN))
            
        if self.processed_audio is not None:
            b64_proc = self._create_spectrogram_base64(self.processed_audio, "Processed Audio Spectrogram")
            self.spectrogram_container.controls.append(ft.Container(
                 ft.Text("Processed Audio Spectrogram", weight="bold"), padding=5))
            self.spectrogram_container.controls.append(ft.Image(src_base64=b64_proc, expand=True, fit=ft.ImageFit.CONTAIN))
            
        self.spectrogram_container.visible = True
        self.page.update()

    def _create_spectrogram_base64(self, audio_data, title):
        fig = Figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        fig.tight_layout()
        # To make it mix well with dark mode:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        return img_b64

    def _play_audio(self, audio_type):
        if audio_type == 'original' and not self.current_audio_path:
            return
        if audio_type == 'processed' and self.processed_audio is None:
            return

        self._stop_audio()
        self.is_playing = True
        self.current_player = audio_type
        
        if audio_type == 'original':
            data, sr = sf.read(self.current_audio_path)
            self.current_time_label = 'original'
        else:
            data = self.processed_audio
            sr = 16000
            self.current_time_label = 'processed'
            
        self.total_duration = int(len(data) / sr)
        self.audio_data = data
        self.sample_rate = sr
        self.current_frame = 0
        
        def callback(outdata, frames, time, status):
            if status:
                pass
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
        
        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            callback=callback,
            finished_callback=self._on_playback_finished
        )
        self.stream.start()
        
        if hasattr(self, 'update_time_thread') and self.update_time_thread:
            self.update_time_thread.join(timeout=1.0)
            
        self.update_time_thread = threading.Thread(target=self._update_time)
        self.update_time_thread.daemon = True
        self.update_time_thread.start()

        # Update icons to indicate playing state
        if audio_type == 'original':
            self.original_play_btn.icon = ft.icons.PAUSE
        else:
            self.processed_play_btn.icon = ft.icons.PAUSE
        self.page.update()

    def _stop_audio(self):
        self.is_playing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if hasattr(self, 'update_time_thread') and self.update_time_thread:
            self.update_time_thread.join(timeout=1.0)
            self.update_time_thread = None
            
        self.current_player = None
        
        self.original_play_btn.icon = ft.icons.PLAY_ARROW
        self.processed_play_btn.icon = ft.icons.PLAY_ARROW
        
        # Reset labels
        if hasattr(self, 'current_time_label'):
            if self.current_time_label == 'original':
                self.original_time.value = f"0:00 / {self.format_time(self.total_duration)}"
            elif self.current_time_label == 'processed':
                self.processed_time.value = f"0:00 / {self.format_time(self.total_duration)}"
            self.page.update()

    def _on_playback_finished(self):
        self.is_playing = False
        self.current_player = None
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        self.original_play_btn.icon = ft.icons.PLAY_ARROW
        self.processed_play_btn.icon = ft.icons.PLAY_ARROW
            
        if hasattr(self, 'current_time_label'):
            if self.current_time_label == 'original':
                self.original_time.value = f"0:00 / {self.format_time(self.total_duration)}"
            elif self.current_time_label == 'processed':
                self.processed_time.value = f"0:00 / {self.format_time(self.total_duration)}"
            try:
                self.page.update()
            except Exception:
                pass


    def _update_time(self):
        last_pos = -1
        while self.is_playing:
            try:
                if self.audio_data is not None and hasattr(self, 'current_frame'):
                    current_pos = int(self.current_frame / self.sample_rate)
                    if current_pos != last_pos:
                        last_pos = current_pos
                        formatted_current = self.format_time(current_pos)
                        formatted_total = self.format_time(self.total_duration)
                        if self.current_time_label == 'original':
                            self.original_time.value = f"{formatted_current} / {formatted_total}"
                        elif self.current_time_label == 'processed':
                            self.processed_time.value = f"{formatted_current} / {formatted_total}"
                        self.page.update()
                time.sleep(0.1)
            except Exception as e:
                break

    def on_file_saved(self, e):
        if e.path and self.processed_audio is not None:
            output_path = e.path
            if not output_path.endswith('.wav'):
                output_path += '.wav'
            try:
                self.audio_processor.save_audio(self.processed_audio, output_path)
                self.show_message("Audio saved successfully")
            except Exception as ex:
                self.show_message(f"Failed to save audio: {str(ex)}", is_error=True)

    def __del__(self):
        self._stop_audio()


def main(page: ft.Page):
    app = NocleGUI(page)

if __name__ == "__main__":
    ft.app(target=main)