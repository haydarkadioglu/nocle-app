import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

from gui import main as start_gui
from model_download import download_model

if __name__ == "__main__":
    # Önce modeli kontrol et ve gerekirse indir
    if download_model():
        import flet as ft
        ft.app(target=start_gui)
    else:
        print("❌ Model indirilemedi. Program başlatılamıyor.")