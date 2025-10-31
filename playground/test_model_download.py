import os
import requests

def download_model():
    # Model dosyasının kaydedileceği klasörü oluştur
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "nocle.hdf5")
    
    # Eğer model zaten varsa, indirme
    if os.path.exists(model_path):
        print("✅ Model zaten mevcut:", model_path)
        return True

    url = "https://huggingface.co/haydarkadioglu/nocle-app/resolve/main/nocle.hdf5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✅ nocle.hdf5 başarıyla indirildi:", model_path)
        return True
    except Exception as e:
        print(f"❌ İndirme hatası: {str(e)}")
        return False

if __name__ == "__main__":
    download_model()


