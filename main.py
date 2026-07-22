import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from model_download import download_model

if __name__ == "__main__":
    if not download_model():
        print("❌ Model indirilemedi. Program başlatılamıyor.")
    else:
        import webview
        from api import Api

        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui', 'index.html')
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