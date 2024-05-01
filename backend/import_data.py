import glob
import rich.progress
import logging
from rich.logging import RichHandler
from rich import print
from talon import signature
from talon.quotation import extract_from_plain

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")

def clean_email(file_path):
    # Read the email content from the file
    log.info(f'Cleaning {file_path}')
    with open(file_path, 'r', encoding='utf-8') as file:
        email_text = file.read()
    print(email_text)

    # Use Talon to extract the body and signature
    body, signature = signature.extract_signature(email_text)

    # Optionally, remove quoted text
    clean_body = extract_from_plain(body)

    return clean_body

log.info('Compiling email files')
email_files = list(glob.glob('./dataset/maildir/**/*.', recursive=True))
email_files[:10]
with rich.progress.Progress() as progress:
    task = progress.add_task('[cyan]Adding emails to vector database...', total=len(email_files))
    for file in email_files:
        email = clean_email(file)
        print(email)
        print()
        progress.update(task, advance=1)
