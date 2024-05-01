import glob
import rich.progress
import logging
import mailparser
import chardet
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

log.info('Compiling email files')
email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))
print(len(email_files))
with rich.progress.Progress() as progress:
    task = progress.add_task('[cyan]Adding emails to vector database...', total=len(email_files))
    for file in email_files:
        email = clean_email(file)
        progress.update(task, advance=1)
