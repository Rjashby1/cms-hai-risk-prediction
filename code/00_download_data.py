import os
import urllib.request
import zipfile
from pathlib import Path

def download_and_extract_cms_data(target_dir="../data/raw/cms_hospitals_jan2026"):
    """
    Checks if the CMS data exists locally. If not, downloads the current 
    hospital data archive directly from CMS and extracts it.
    """
    current_dir = Path(__file__).parent
    data_path = (current_dir / target_dir).resolve()
    
    # The direct, stable URL for the latest CMS hospital data archive
    cms_url = "https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/current/hospitals_current_data.zip"
    zip_path = data_path / "hospitals_current_data.zip"
    
    # Check if a known file already exists to avoid re-downloading
    check_file = data_path / "Healthcare_Associated_Infections-Hospital.csv"
    
    if check_file.exists():
        print(f"[SKIP] Data already exists in {data_path}")
        return True

    print(f"Data not found locally. Preparing to download...")
    
    # Create the directory if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    print(f"Downloading ~100MB archive from CMS...\nURL: {cms_url}")
    try:
        # A simple progress hook to show it hasn't frozen
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                # Print progress on the same line
                print(f"\rProgress: {percent:.1f}%", end="")
                
        urllib.request.urlretrieve(cms_url, zip_path, reporthook=show_progress)
        print("\n[SUCCESS] Download complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False

    # Extract the zip file
    print("Extracting CSV files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"[SUCCESS] Extracted all files to {data_path}")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False
        
    # Clean up the heavy zip file to save space
    print("Cleaning up zip file...")
    os.remove(zip_path)
    print("[SUCCESS] Data pipeline ready.")
    
    return True

if __name__ == "__main__":
    download_and_extract_cms_data()