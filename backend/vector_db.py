import faiss
import json
import numpy as np

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

class database():
    def __init__(faiss_index_path, file_ids_path):
        self.file_ids = json.load(open('file_ids.json', 'r'))
        self.faiss_index = faiss.read_index('index.faiss')


log = logging.getLogger("rich")
if __name__ == "__main__":
    file_ids = json.load(open('file_ids.json', 'r'))
    faiss_index = faiss.read_index('index.faiss')
