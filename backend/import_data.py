import glob
import logging
import mailparser
import chardet
import re
import datetime
from rich.progress import track
from rich.logging import RichHandler
from rich import print

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.ERROR,
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
    log.info(f'Cleaning {file_path}')
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
        date_str = date_str.split(' (')[0]
        return datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    except ValueError as e:
        log.warning(f"Date parsing error: {e}")
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

    return metadata

def get_metadata(file_path):
    log.info(f'Getting metadata for {file_path}')
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            email_text = file.read()
        metadata = extract_metadata(email_text)
        return metadata
    except Exception as e:
        log.exception(f'Error processing {file_path}: {e}')
        return None


log.info('Compiling email files')
email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))
email_files = email_files[:50]
for file in track(email_files, description='[cyan]Adding emails to vector database...'):
    email = clean_email(file)
    metadata = get_metadata(file)
    print(metadata)
