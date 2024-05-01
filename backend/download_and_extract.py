import os
import requests
import tarfile
import logging
from rich.logging import RichHandler
from rich.progress import Progress

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")


def download_file(url: str, save_path: str) -> None:
    """Downloads a file from a specified URL and saves it to the given path with progress tracking."""
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Checks for request errors
            total_size_in_bytes = int(
                response.headers.get('content-length', 0))
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Downloading {save_path}...", total=total_size_in_bytes)
                with open(save_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        progress.update(task, advance=len(chunk))
        log.info(f"File downloaded successfully: {save_path}")
    except Exception as e:
        log.error(f"Failed to download the file: {e}")
        raise


def extract_tar_gz(file_path: str, extract_to: str) -> None:
    """Extracts a tar.gz file to a specified directory with progress tracking."""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            members = tar.getmembers()
            total_members = len(members)
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Extracting {file_path}...", total=total_members)
                for member in members:
                    tar.extract(member, path=extract_to)
                    progress.update(task, advance=1)
        log.info(f"Extracted to {extract_to}")
    except Exception as e:
        log.error(f"Failed to extract the file: {e}")
        raise


if __name__ == "__main__":
    dataset_url: str = 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz'
    compressed_file: str = './enron_email_dataset.tar.gz'
    directory: str = './dataset/'

    if not os.path.exists(compressed_file):
        log.info("Downloading Enron email dataset...")
        download_file(dataset_url, compressed_file)
    else:
        log.info(
            f"File {compressed_file} already exists, skipping download.")

    if not os.path.isdir(directory):
        log.info(f"Extracting Enron email dataset to {directory}...")
        extract_tar_gz(compressed_file, directory)
        # Optionally remove the tar.gz file after extraction
        os.remove(compressed_file)
        log.info(f"Deleted downloaded file {compressed_file}")
    else:
        log.info("Enron email dataset already extracted.")
