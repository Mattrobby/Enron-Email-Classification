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

from dateutil import parser
from rich import print
from rich.logging import RichHandler
from rich.progress import track
from sentence_transformers import SentenceTransformer

# Setup rich handler for better logging experience
logging.getLogger("transformers").setLevel(logging.WARNING)
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
        return email.body
    except Exception as e:
        log.exception(f'Error processing {file_path}: {e}')
        return None

def parse_date(date_str):
    try:
        # Strip off common trailing issues such as 'GMT' and 'Subject:'
        clean_str = re.sub(r'(\sGMT)?\s*Subject:.*', '', date_str)
        return parser.parse(clean_str)
    except ValueError as e:
        log.warning(f"Date parsing error: {e} for date string: {date_str}")
        return None

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
                    metadata[current_key] = parse_date(value)
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

    num_to_process = 1000
    files_to_process = random.sample(email_files, num_to_process)

    log.info('Adding emails to vector database')
    file_ids = {}
    for index, file in enumerate(track(files_to_process, description='[cyan]Adding emails to vector database...')):
        email = clean_email(file)
        metadata = get_metadata(file)
        metadata['File'] = file
        file_ids[index] = metadata

        if email is not None:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            vector = model.encode([email])[0]
            faiss_index.add(np.array([vector]))

            sys.stdout = original_stdout


    return faiss_index, file_ids

if __name__ == "__main__":
    create_vector_db()
