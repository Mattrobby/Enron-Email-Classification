"""
File: import_data.py
Description: Downloads Enron email dataset, extracts metadata, and stores it in a Weaviate collection.
Author: Matthew Stefanovic
Email: matthew@stefanovic.us
Created: 2024-07-14

Changelog:
    Version 1.0.0 - 2024-07-14 - Matthew Stefanovic
        - Migraded code from download_and_extract.py to this file
    Version 1.0.1 - 2024-07-19 - Matthew Stefanovic
        - Imported code from import_data.py that didn't use Faiss
"""
import os
import requests
import tarfile
import chardet
import faiss
import glob
import logging
import mailparser
import numpy as np
import random
import re
import sqlite3
import threading

from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from sentence_transformers import SentenceTransformer
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")


# =======================================
# Download and Extract Data
# =======================================

def download_file(url: str, save_path: str) -> None:
    """Downloads a file from a specified URL and saves it to the given path with progress tracking."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Checks for request errors
            total_size_in_bytes = int(
                response.headers.get('content-length', 0))
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Downloading Enron Email Dataset...", total=total_size_in_bytes)
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


# TODO: add in a hash and logic so it checks if the file has already been downloaded and extracted
def download_and_extract() -> None:
    dataset_url: str = 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz'
    compressed_file: str = './dataset/enron_email_dataset.tar.gz'
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


# =======================================
# Import Data
# =======================================
def detect_encoding(file_path) -> str:
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']


def clean_email(file_path) -> (str | None, str | None):
    log.debug(f'Cleaning {file_path}')
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            email_text = file.read()
        email_text = email_text.replace('\\', '\\\\')
        email = mailparser.parse_from_string(email_text)
        return email.body, email_text
    except Exception as e:
        return None, None


def parse_email_list(emails) -> str | None:
    if emails:
        emails = re.sub(r'\s+', ' ', emails.replace('\n', ''))
        return [email.strip() for email in emails.split(',')]
    return None


def parse_complex_field(field_value) -> str:
    parts = {}
    for item in field_value.split('; '):
        if '=' in item:
            key, value = item.split('=', 1)
            parts[key.strip()] = value.strip()
        else:
            parts[item.strip()] = None
    if parts:
        return {'type': field_value.split(';')[0].strip(), **parts}
    return field_value.strip()


def extract_metadata(email_text) -> dict:
    metadata = {}
    current_key = None
    buffer = []

    for line in email_text.splitlines():
        line = line.strip()
        if ': ' in line:
            if current_key:
                value = ' '.join(buffer).strip()
                if current_key in ['Date']:
                    metadata[current_key] = value
                elif current_key in ['To', 'Cc', 'Bcc']:
                    metadata[current_key] = parse_email_list(value)
                elif current_key in ['Content-Type']:
                    metadata[current_key] = parse_complex_field(value)
                else:
                    metadata[current_key] = value
            current_key, value = line.split(': ', 1)
            buffer = [value]
        else:
            buffer.append(line)

    return metadata


def get_metadata(file_path) -> dict | None:
    log.debug(f'Getting metadata for {file_path}')
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            lines = []
            for _ in range(15):
                line = file.readline()
                if not line:
                    break
                lines.append(line)
            email_text = ''.join(lines)
        metadata = extract_metadata(email_text)
        return metadata
    except Exception as e:
        log.exception(f'Error processing {file_path}: {e}')
        return None

