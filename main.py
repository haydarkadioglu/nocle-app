import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from model_download import download_model
import setup_checker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, 'icon.png')

def launch_main_app():
    import webview
    from api import Api

    ui_path = os.path.join(BASE_DIR, 'ui', 'index.html')
    api = Api()

    window = webview.create_window(
        title='Nocle — Audio Enhancer',
        url=f'file:///{ui_path.replace(os.sep, "/")}',
        js_api=api,
        width=1020,
        height=680,
        min_size=(780, 520),
        background_color='#0d0d0d',
    )
    api.set_window(window)
    webview.start(debug=False)

def launch_setup():
    import webview
    from setup_api import SetupApi

    setup_ui_path = os.path.join(BASE_DIR, 'ui', 'setup.html')
    setup_api = SetupApi(on_done_callback=launch_main_app)

    window = webview.create_window(
        title='Nocle — Setup Wizard',
        url=f'file:///{setup_ui_path.replace(os.sep, "/")}',
        js_api=setup_api,
        width=600,
        height=620,
        resizable=False,
        background_color='#0d0d0d',
    )
    setup_api.set_window(window)
    webview.start(debug=False)

if __name__ == "__main__":
    if not download_model():
        print("❌ Model indirilemedi. Program başlatılamıyor.")
    else:
        if not setup_checker.all_deps_ok():
            launch_setup()
        else:
            launch_main_app()
