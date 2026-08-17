from pathlib import Path
import re
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Adjust this based on your project structure

def get_all_files(base_path: str, file_extension: str) -> list:
    """
    Get all files with the specified extension in the given base path.

    Args:
        base_path (str): The base directory to search for files.
        file_extension (str): The file extension to filter by (e.g., '.zip').

    Returns:
        list: A list of file paths matching the specified extension.
    """
    base_path = Path(base_path)
    print(f"Searching for files in: {base_path} with extension: {file_extension}")
    return [str(file) for file in base_path.rglob(f"*{file_extension}")]


def get_month_file(base_path: str, year: str, month: str, file_extension: str = ".npy") -> Path:
    """
    Get the traffic file for a specific year and month.

    Args:
        base_path (str): Project base directory.
        year (str): Year, e.g. "2023".
        month (str): Month, e.g. "01" or "1".
        file_extension (str): File extension, e.g. ".npy".

    Returns:
        Path: Path to the requested monthly file.
    """

    month = f"{int(month):02d}"

    file_path = Path(base_path) / "Data" / "archive" / f"year_{year}" / f"{year}_p{month}{file_extension}"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Traffic file not found: {file_path}"
        )

    return file_path

def get_year_file(base_path: str, year: str, file_extension: str = ".npy") -> Path:
    """
    Get the traffic file for a specific year.

    Args:
        base_path (str): Project base directory.
        year (str): Year, e.g. "2023".
        file_extension (str): File extension, e.g. ".npy".

    Returns:
        Path: Path to the requested yearly file.
    """
    file_path = Path(base_path) / "Data" / "archive" / f"year_{year}"
    print(f"Searching for files in: {file_path} with extension: {file_extension}")
    if not file_path.exists():
        raise FileNotFoundError(
            f"Traffic file not found: {file_path}"
        )
    files = [f for f in file_path.rglob(f"*{file_extension}") if f.is_file()]
    sorted_files = sorted(files, key=lambda x: x.name)
    
    
    return sorted_files

print(get_year_file(BASE_DIR, "2023", ".npy"))