from pathlib import Path
import shutil
import zipfile
import sys

# Optional: if you want to hardcode the source zip path, put it here.
# Example:
# ZIP_SOURCE = Path(r"D:\Downloads\hospitals_current_data.zip")
ZIP_SOURCE = None

ZIP_NAME = "hospitals_current_data.zip"
EXTRACTED_FOLDER_NAME = "cms_hospitals_jan2026"

KEY_FILES = [
    "Healthcare_Associated_Infections-Hospital.csv",
    "HCAHPS-Hospital.csv",
    "Timely_and_Effective_Care-Hospital.csv",
    "Hospital_General_Information.csv",
    "FY_2026_HAC_Reduction_Program_Hospital.csv",
    "HOSPITAL_Data_Dictionary.pdf",
    "Measure_Dates.csv",
    "Data_Updates_January_2026.csv",
]


def choose_zip_file() -> Path | None:
    """Open a file picker so the user can choose the CMS zip."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title="Select hospitals_current_data.zip",
        filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
    )

    root.destroy()
    return Path(selected) if selected else None


def find_repo_root() -> Path:
    """
    Assumes this script is saved in the repo root.
    """
    return Path(__file__).resolve().parent


def copy_zip(source_zip: Path, destination_zip: Path) -> None:
    destination_zip.parent.mkdir(parents=True, exist_ok=True)

    if source_zip.resolve() == destination_zip.resolve():
        print(f"[INFO] ZIP is already in the correct location:\n  {destination_zip}")
        return

    shutil.copy2(source_zip, destination_zip)
    print(f"[OK] Copied ZIP to:\n  {destination_zip}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    print(f"[OK] Extracted ZIP to:\n  {extract_to}")


def report_key_files(extract_to: Path) -> None:
    print("\nChecking for key project files...\n")

    found = []
    missing = []

    for name in KEY_FILES:
        matches = list(extract_to.rglob(name))
        if matches:
            found.append(matches[0])
            print(f"[FOUND] {name}")
        else:
            missing.append(name)
            print(f"[MISSING] {name}")

    print("\nSummary:")
    print(f"  Found:   {len(found)}")
    print(f"  Missing: {len(missing)}")

    if found:
        print("\nResolved paths:")
        for path in found:
            print(f"  - {path}")

    if missing:
        print("\nMissing files:")
        for name in missing:
            print(f"  - {name}")


def write_inventory(extract_to: Path) -> None:
    inventory_path = extract_to / "_file_inventory.txt"
    files = sorted([p for p in extract_to.rglob("*") if p.is_file()])

    with inventory_path.open("w", encoding="utf-8") as f:
        for file_path in files:
            rel = file_path.relative_to(extract_to)
            f.write(str(rel) + "\n")

    print(f"\n[OK] Wrote inventory file:\n  {inventory_path}")


def main() -> None:
    repo_root = find_repo_root()
    data_raw = repo_root / "data" / "raw"
    destination_zip = data_raw / ZIP_NAME
    extract_to = data_raw / EXTRACTED_FOLDER_NAME

    print(f"Repo root:\n  {repo_root}")
    print(f"Raw data directory:\n  {data_raw}")

    source_zip = ZIP_SOURCE

    if source_zip is None:
        source_zip = choose_zip_file()

    if source_zip is None:
        print("\n[ERROR] No ZIP file selected.")
        sys.exit(1)

    source_zip = Path(source_zip)

    if not source_zip.exists():
        print(f"\n[ERROR] ZIP file not found:\n  {source_zip}")
        sys.exit(1)

    if source_zip.suffix.lower() != ".zip":
        print(f"\n[ERROR] Selected file is not a .zip:\n  {source_zip}")
        sys.exit(1)

    copy_zip(source_zip, destination_zip)
    extract_zip(destination_zip, extract_to)
    report_key_files(extract_to)
    write_inventory(extract_to)

    print("\nDone.")
    print("Your raw CMS data are now stored locally under data/raw/ and should remain ignored by Git.")


if __name__ == "__main__":
    main()