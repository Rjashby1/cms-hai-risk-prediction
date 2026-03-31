import os
import urllib.request
import zipfile
from pathlib import Path


def download_and_extract_cms_data(
    target_dir="../data/raw/cms_hospitals_jan2026",
    check_file_name="Healthcare_Associated_Infections-Hospital.csv",
):
    """
    Ensure that the CMS hospital downloadable database is available locally.

    This function checks whether a known CSV file already exists in the target
    directory. If the file is present, the download step is skipped. If not,
    the function downloads the current CMS hospital archive, extracts its
    contents, removes the zip file, and returns a success flag along with the
    resolved data directory.

    Parameters
    ----------
    target_dir : str, default="../data/raw/cms_hospitals_jan2026"
        Relative path to the local raw-data directory.
    check_file_name : str, default="Healthcare_Associated_Infections-Hospital.csv"
        Filename used to determine whether the CMS archive has already been
        downloaded and extracted.

    Returns
    -------
    tuple
        (success_flag, resolved_data_path)
    """
    current_dir = Path(__file__).parent
    data_path = (current_dir / target_dir).resolve()

    cms_url = (
        "https://data.cms.gov/provider-data/sites/default/files/archive/"
        "Hospitals/current/hospitals_current_data.zip"
    )
    zip_path = data_path / "hospitals_current_data.zip"
    check_file = data_path / check_file_name

    if check_file.exists():
        print(f"[SKIP] CMS data already exists at: {data_path}")
        return True, data_path

    print("[INFO] CMS data not found locally.")
    print(f"[INFO] Preparing download target: {data_path}")

    data_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading CMS hospital archive...")
    print(f"[INFO] Source URL: {cms_url}")

    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r[DOWNLOAD] Progress: {percent:5.1f}%", end="")

        urllib.request.urlretrieve(cms_url, zip_path, reporthook=show_progress)
        print("\n[SUCCESS] Download complete.")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False, data_path

    print("[INFO] Extracting archive contents...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        print(f"[SUCCESS] Extracted all files to: {data_path}")

    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False, data_path

    print("[INFO] Removing downloaded zip file...")

    try:
        if zip_path.exists():
            os.remove(zip_path)
        print("[SUCCESS] Zip cleanup complete.")

    except Exception as e:
        print(f"[WARNING] Extraction succeeded, but zip cleanup failed: {e}")

    print("[SUCCESS] CMS raw data is ready.")
    return True, data_path


if __name__ == "__main__":
    success, resolved_path = download_and_extract_cms_data()
    print(f"Success: {success}")
    print(f"Path: {resolved_path}")