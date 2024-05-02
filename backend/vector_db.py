import faiss
import tiktoken
import random
import json
import logging
import numpy as np
import os 

from dotenv import load_dotenv
from openai import OpenAI
from rich import print
from sklearn.cluster import KMeans
from rich.logging import RichHandler
from rich.progress import track

load_dotenv()
OPEN_AI_APIKEY = os.environ.get('OPEN_AI_APIKEY')

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

class faiss_database():
    def __init__(self, faiss_index_path, file_ids_path):
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.file_ids = json.load(open(file_ids_path, 'r'))
        self.model = 'llama3'
        self.client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key=OPEN_AI_APIKEY, # required, but unused
        )

    def display_clusters(self, clustered_emails, cluster_descriptions):
        clusters = ''
        for cluster_id, emails in clustered_emails.items():
            clusters += f'Cluster {cluster_id+1}: "{cluster_descriptions[cluster_id]}"\t({len(emails)} emails)\n'
        log.info(clusters)

    def generate_category_descriptions(self, clustered_emails, sample_size=30):
        cluster_descriptions = {}
        system_message = "Identify a single-word theme for the list of emails. Respond only with the category, do not include explanations"
        for cluster_id, emails in clustered_emails.items():
            sample_emails = random.sample(emails, min(sample_size, len(emails)))
            combined_text = " ".join([email['Email Body'] for email in sample_emails if email['Email Body']])
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{combined_text}\n"}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=8,
                temperature=0.5,
                top_p=0.5,
            )

            description = response.choices[0].message.content.strip()
            cluster_descriptions[cluster_id] = description
            for email in emails:
                email['Cluster Description'] = description

        return clustered_emails, cluster_descriptions

    def cluster(self, num_clusters=10):
        # Step 1: Extract vectors from the FAISS index
        log.info('Extract vectors from the FAISS index')
        vectors = np.zeros((self.faiss_index.ntotal, self.faiss_index.d), dtype='float32')
        self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal, vectors)

        # Step 2: Clustering with K-means
        log.info('Clustering with K-means')
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(vectors)

        # Step 3: Assign emails to clusters
        log.info('Assign emails to clusters')
        clustered_emails = {i: [] for i in range(num_clusters)}
        for idx, cluster_id in enumerate(track(clusters, description='[cyan]Clustering emails...')):
            email_info = self.file_ids[str(idx)]  # Ensure the keys are aligned with indices
            clustered_emails[cluster_id].append(email_info)

        # Step 4: Catagorize Clusters
        clustered_emails, clustered_descriptions = self.generate_category_descriptions(clustered_emails)
        self.display_clusters(clustered_emails, clustered_descriptions)
        
        return clustered_emails, clustered_descriptions

log = logging.getLogger("rich")
if __name__ == "__main__":
    vector_db = faiss_database('index.faiss', 'file_ids.json')
    cluster = vector_db.cluster(num_clusters=5)
