import chardet
import faiss
import glob
import logging
import mailparser
import numpy as np 
import random
import re
import os
import sys
import json

from rich import print
from rich.logging import RichHandler
from sentence_transformers import SentenceTransformer
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def clean_email(file_path):
    log.debug(f'Cleaning {file_path}')
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            email_text = file.read()
        email_text = email_text.replace('\\', '\\\\')
        email = mailparser.parse_from_string(email_text)
        return email.body, email_text
    except Exception as e:
        log.exception(f'Error processing {file_path}: {e}')
        return None, None

def parse_email_list(emails):
    if emails:
        emails = re.sub(r'\s+', ' ', emails.replace('\n', ''))
        return [email.strip() for email in emails.split(',')]
    return None

def parse_complex_field(field_value):
    parts = {}
    for item in field_value.split('; '):
        if '=' in item:
            key, value = item.split('=', 1)  # Split at the first occurrence of '='
            parts[key.strip()] = value.strip()
        else:
            parts[item.strip()] = None  # Handle cases where there is no '=' to define a value
    if parts:
        return {'type': field_value.split(';')[0].strip(), **parts}
    return field_value.strip()

def extract_metadata(email_text):
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

def get_metadata(file_path):
    log.debug(f'Getting metadata for {file_path}')
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            # Read the first 15 lines
            lines = []
            for _ in range(15):
                line = file.readline()
                if not line:
                    break  # End of file reached
                lines.append(line)
            email_text = ''.join(lines)
        metadata = extract_metadata(email_text)
        return metadata
    except Exception as e:
        log.exception(f'Error processing {file_path}: {e}')
        return None


def create_vector_db():
    log.info('Compiling email files')
    email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize a FAISS index
    log.info('Initialize a FAISS index')
    d = 384  # Dimension of vectors, change based on the model used
    faiss_index = faiss.IndexFlatL2(d)  # Using IndexFlatL2 for simplicity

    num_to_process = 100000
    files_to_process = random.sample(email_files, num_to_process)

    log.info('Adding emails to vector database')
    file_ids = {}
    with Progress(
        TextColumn("[bold cyan]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TextColumn("[bold yellow]({task.percentage:.0f}%)"),
        TimeRemainingColumn(),  # Estimates the time remaining
        expand=True
    ) as progress:
        task = progress.add_task("[cyan]Adding emails to vector database...", total=len(files_to_process))
        for index, file in enumerate(files_to_process):
            email_body, email_text = clean_email(file)
            metadata = get_metadata(file)
            metadata['File'] = file
            metadata['Email Body'] = email_body
            metadata['Email Text'] = email_text
            file_ids[index] = metadata

            if email_body is not None:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

                vector = model.encode([email_body])[0]
                faiss_index.add(np.array([vector]))

                sys.stdout = original_stdout

            progress.update(task, advance=1)

    faiss.write_index(faiss_index, 'index.faiss')
    with open('file_ids.json', 'w') as f:
        json.dump(file_ids, f)

    return faiss_index, file_ids

if __name__ == "__main__":
    create_vector_db()
