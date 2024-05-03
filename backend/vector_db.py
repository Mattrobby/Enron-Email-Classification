import faiss
import tiktoken
import sqlite3
import random
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
    def __init__(self, faiss_index_path, db_path):
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.db_path = db_path
        self.model = 'gpt-3.5-turbo'
        self.max_tokens = 16385
        self.client = OpenAI(
            api_key=OPEN_AI_APIKEY,
        )

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    def fetch_emails(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT email_index, email_body FROM emails")
            return {row[0]: {'Email Body': row[1]} for row in cursor.fetchall()}

    def display_clusters(self, clustered_emails, cluster_descriptions):
        clusters = ''
        for cluster_id, emails in clustered_emails.items():
            clusters += f'Cluster {cluster_id+1}: "{cluster_descriptions[cluster_id]}"\t({len(emails)} emails)\n'
        log.info(clusters)

    def cluster_email_sample(self, sample_emails):
        return " ".join([email['Email Body'] for email in sample_emails if email['Email Body']])

    def generate_category_descriptions(self, clustered_emails, sample_size=30):
        cluster_descriptions = {}
        system_message = "Identify a single-word theme for the list of emails. Respond only with the category, do not include explanations"
        adjustment_attempts = 15  # Limit the number of adjustments to prevent infinite loops
    
        for cluster_id, emails in clustered_emails.items():
            if len(emails) == 0:
                logging.warning(f"No emails in cluster {cluster_id}")
                continue
    
            sample_emails = random.sample(emails, min(sample_size, len(emails)))
            combined_text = self.cluster_email_sample(sample_emails)
    
            for _ in range(adjustment_attempts):
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": combined_text}
                ]
                encoding = tiktoken.encoding_for_model(self.model)
                token_count = len(encoding.encode(str(messages)))
    
                if token_count > self.max_tokens:
                    if len(sample_emails) > 1:
                        sample_emails.pop()
                        combined_text = self.cluster_email_sample(sample_emails)
                        logging.warning(f'Reduced sample size due to token limit. New token count: {token_count}')
                        if adjustment_attempts == 0:
                            adjustment_attempts += 1
                    else:
                        logging.error("Minimum sample size still exceeds token limit")
                        break
                elif token_count < (self.max_tokens * 0.75) and len(sample_emails) < len(emails):
                    new_sample_email = random.choice(emails)
                    sample_emails.append(new_sample_email)
                    combined_text = self.cluster_email_sample(sample_emails)
                    logging.info(f'Increased sample size. New token count: {token_count}')
                else:
                    logging.info(f'Sample size is optimal. Token count: {token_count}')
                    break

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
        # Extract vectors from the FAISS index
        log.info('Extract vectors from the FAISS index')
        vectors = np.zeros((self.faiss_index.ntotal, self.faiss_index.d), dtype='float32')
        self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal, vectors)

        # Clustering with K-means
        log.info('Clustering with K-means')
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(vectors)

        # Assign emails to clusters
        log.info('Assign emails to clusters')
        emails = self.fetch_emails()
        clustered_emails = {i: [] for i in range(num_clusters)}
        for idx, cluster_id in enumerate(track(clusters, description='[cyan]Clustering emails...')):
            if idx in emails:  # Check if the index exists in the fetched emails
                email_info = emails[idx]
                clustered_emails[cluster_id].append(email_info)

        # Categorize Clusters
        clustered_emails, cluster_descriptions = self.generate_category_descriptions(clustered_emails)
        self.display_clusters(clustered_emails, cluster_descriptions)
        
        return clustered_emails, cluster_descriptions

log = logging.getLogger("rich")
if __name__ == "__main__":
    vector_db = faiss_database('index.faiss', 'emails.db')
    cluster = vector_db.cluster(num_clusters=5)
