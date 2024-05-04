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
from concurrent.futures import ThreadPoolExecutor, as_completed


# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
index_lock = Lock()

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
    cursor.execute("PRAGMA journal_mode=WAL;")
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

def insert_metadata(metadata_batch, db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    values_to_insert = [
        (
            metadata.get('File'), metadata.get('Message-ID'), metadata.get('Date'), metadata.get('From'),
            ','.join(metadata.get('To', [])), metadata.get('Subject'), metadata.get('Mime-Version'),
            metadata.get('Content-Type', {}).get('type'), metadata.get('Content-Transfer-Encoding'),
            metadata.get('X-From'), metadata.get('X-To'), metadata.get('X-Folder'), metadata.get('X-Origin'),
            metadata.get('Email Body'), metadata.get('Email Text'), metadata.get('Email Index')
        )
        for metadata in metadata_batch
    ]
    try:
        cursor.executemany('''
            INSERT INTO emails (
                file, message_id, date, from_email, to_emails, subject, mime_version, content_type,
                content_transfer_encoding, x_from, x_to, x_folder, x_origin, email_body, email_text, email_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values_to_insert)
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.exception(f"Failed to insert batch: {e}")
    finally:
        cursor.close()
        conn.close()

def process_email_batch(email_body_batch, metadata_batch, faiss_index, db_path, model, progress):
    log.info('Vectorizing emails')
    vectors = model.encode(email_body_batch, show_progress_bar=False)
    log.info('Adding vectors to Faiss')
    with index_lock:
        faiss_index.add(np.array(vectors))

    log.info('Adding to SQLite')
    task = progress.add_task("[red]Adding metadata to SQLite...", total=len(email_body_batch))
    batch_size = 1000
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(0, len(metadata_batch), batch_size):
            batch = metadata_batch[i:i+batch_size]
            future = executor.submit(insert_metadata, batch, db_path)
            futures.append(future)

        for future in as_completed(futures):
            progress.update(task, advance=batch_size)

def create_vector_db():
    log.info('Compiling email files')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))

    log.info('Initialize a FAISS index')
    d = 384
    faiss_index = faiss.IndexFlatL2(d)

    # num_to_process = len(email_files)
    num_to_process = 100000
    files_to_process = random.sample(email_files, num_to_process)

    db_path = 'emails.db'  # Specify the database path
    conn = setup_database()  # Ensure the database is set up before spawning threads
    conn.close()

    log.info('Adding emails to vector database')
    email_body_batch = []
    metadata_batch = []
    batch_size = 10000
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
                metadata['Email Index'] = index
                metadata_batch.append(metadata)

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
