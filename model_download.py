import os
import requests

def download_model():
    # Create directory for model if it doesn't exist
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "nocle.hdf5")
    
    # Skip download if model already exists
    if os.path.exists(model_path):
        print("✅ Model already exists:", model_path)
        return True

    # Use LFS URL format and proper headers for binary file download
    url = "https://huggingface.co/haydarkadioglu/nocle-app/resolve/main/nocle.hdf5"
    headers = {
        'Accept': 'application/octet-stream',
        'User-Agent': 'Mozilla/5.0'
    }
    
    try:
        # Stream the response to handle large files
        response = requests.get(
            url, 
            headers=headers, 
            stream=True,
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Download the file in chunks to handle large files
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        # Verify file was downloaded successfully
        if os.path.getsize(model_path) > 0:
            print("✅ Model downloaded successfully:", model_path)
            return True
        else:
            raise Exception("Downloaded file is empty")
            
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)  # Remove failed download
        return False