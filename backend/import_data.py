import chardet
import faiss
import glob
import logging
import mailparser
import numpy as np
import os
import random
import re
import sqlite3
import sys
import threading

from rich import print
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from sentence_transformers import SentenceTransformer
from threading import Lock

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
            key, value = item.split('=', 1)
            parts[key.strip()] = value.strip()
        else:
            parts[item.strip()] = None
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

def setup_database():
    conn = sqlite3.connect('emails.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY,
            file TEXT,
            message_id TEXT,
            date TEXT,
            from_email TEXT,
            to_emails TEXT,
            subject TEXT,
            mime_version TEXT,
            content_type TEXT,
            content_transfer_encoding TEXT,
            x_from TEXT,
            x_to TEXT,
            x_folder TEXT,
            x_origin TEXT,
            email_body TEXT,
            email_text TEXT,
            email_index INTEGER
        )
    ''')
    conn.commit()
    return conn

def insert_metadata(conn, metadata, email_index, max_retries=3):
    for attempt in range(max_retries):
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO emails (
                    file, message_id, date, from_email, to_emails, subject, mime_version, content_type,
                    content_transfer_encoding, x_from, x_to, x_folder, x_origin, email_body, email_text, email_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.get('File'), metadata.get('Message-ID'), metadata.get('Date'), metadata.get('From'),
                ','.join(metadata.get('To', [])), metadata.get('Subject'), metadata.get('Mime-Version'),
                metadata.get('Content-Type', {}).get('type'), metadata.get('Content-Transfer-Encoding'),
                metadata.get('X-From'), metadata.get('X-To'), metadata.get('X-Folder'), metadata.get('X-Origin'),
                metadata.get('Email Body'), metadata.get('Email Text'), email_index
            ))
            conn.commit()
            cursor.close()
            break  # Exit the loop if commit is successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            conn.rollback()
            cursor.close()
            if attempt == max_retries - 1:
                print("Max retries reached, failed to insert metadata.")

def process_email_batch(email_body_batch, metadata_batch, faiss_index, db_path, model, progress):
    conn = sqlite3.connect(db_path, check_same_thread=False)  # Each thread gets its own connection
    task = progress.add_task("[red]Adding emails to vector database...", total=len(email_body_batch))
    try:
        index_lock = Lock()
        vectors = model.encode(email_body_batch)
        for vec, (meta, idx) in zip(vectors, metadata_batch):
            faiss_index.add(np.array([vec]))
            with index_lock:
                insert_metadata(conn, meta, idx)
            progress.update(task, advance=1)
    finally:
        conn.close()

def create_vector_db():
    log.info('Compiling email files')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))

    log.info('Initialize a FAISS index')
    d = 384
    faiss_index = faiss.IndexFlatL2(d)

    num_to_process = len(email_files)
    # num_to_process = 50000
    files_to_process = random.sample(email_files, num_to_process)

    db_path = 'emails.db'  # Specify the database path
    setup_database()  # Ensure the database is set up before spawning threads

    log.info('Adding emails to vector database')
    email_body_batch = []
    metadata_batch = []
    batch_size = 50000
    threads = []

    with Progress(
        TextColumn("[bold cyan]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TextColumn("[bold yellow]({task.percentage:.0f}%)"),
        TimeRemainingColumn(),
        expand=True,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Gathering email content...", total=len(files_to_process))

        for index, file in enumerate(files_to_process):
            email_body, email_text = clean_email(file)
            if email_body:
                email_body_batch.append(email_body)
                metadata = get_metadata(file)
                metadata['File'] = file
                metadata['Email Body'] = email_body
                metadata['Email Text'] = email_text
                metadata_batch.append((metadata, index))

            if len(email_body_batch) >= batch_size:
                # Start a new thread to process the batch
                thread = threading.Thread(target=process_email_batch, args=(email_body_batch.copy(), metadata_batch.copy(), faiss_index, db_path, model, progress))
                thread.start()
                threads.append(thread)  # Keep track of the thread
                email_body_batch = []
                metadata_batch = []

            progress.update(task, advance=1)

        # Process any remaining emails after the loop
        if email_body_batch:
            thread = threading.Thread(target=process_email_batch, args=(email_body_batch, metadata_batch, faiss_index, db_path, model, progress))
            thread.start()
            threads.append(thread)

# Wait for all threads to complete
    for thread in threads:
        thread.join()

    faiss.write_index(faiss_index, 'index.faiss')

    return faiss_index

if __name__ == "__main__":
    create_vector_db()
