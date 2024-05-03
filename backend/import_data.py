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
import sqlite3

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
    conn = sqlite3.connect('emails.db')
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

def insert_metadata(conn, metadata, email_index):
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

def create_vector_db():
    log.info('Compiling email files')
    email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))
    model = SentenceTransformer('all-MiniLM-L6-v2')

    log.info('Initialize a FAISS index')
    d = 384
    faiss_index = faiss.IndexFlatL2(d)

    num_to_process = 1000
    files_to_process = random.sample(email_files, num_to_process)

    conn = setup_database()

    log.info('Adding emails to vector database')
    with Progress(
        TextColumn("[bold cyan]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TextColumn("[bold yellow]({task.percentage:.0f}%)"),
        TimeRemainingColumn(),
        expand=True
    ) as progress:
        task = progress.add_task("[cyan]Adding emails to vector database...", total=len(files_to_process))
        for index, file in enumerate(files_to_process):
            email_body, email_text = clean_email(file)
            metadata = get_metadata(file)
            metadata['File'] = file
            metadata['Email Body'] = email_body
            metadata['Email Text'] = email_text

            insert_metadata(conn, metadata, index)

            if email_body is not None:
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

                vector = model.encode([email_body])[0]
                faiss_index.add(np.array([vector], dtype='float32'))

                sys.stdout = original_stdout

            progress.update(task, advance=1)

    faiss.write_index(faiss_index, 'index.faiss')
    conn.close()

    return faiss_index

if __name__ == "__main__":
    create_vector_db()
