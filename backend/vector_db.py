import faiss
import tiktoken
import sqlite3
import random
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from rich.logging import RichHandler
from rich.progress import track
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

load_dotenv()
OPEN_AI_APIKEY = os.environ.get('OPEN_AI_APIKEY')

# Setup rich handler for better logging experience
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

class Database():
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
        return " ".join([email['Email Info']['Email Body'] for email in sample_emails if email['Email Info'].get('Email Body')])

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
    
            index = 0
            while index < adjustment_attempts:
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
                    else:
                        logging.error("Minimum sample size still exceeds token limit")
                        break
                elif token_count < (self.max_tokens * 0.75) and len(sample_emails) < len(emails):
                    new_sample_email = random.choice(emails)
                    sample_emails.append(new_sample_email)
                    combined_text = self.cluster_email_sample(sample_emails)
                    logging.info(f'Increased sample size. New token count: {token_count}')
                    index += 1
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
                email['Email Info']['Cluster Description'] = description

        return clustered_emails, cluster_descriptions

    def ensure_column(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(emails)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'category' not in columns:
                cursor.execute("ALTER TABLE emails ADD COLUMN category TEXT")

    def ensure_index(self):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            # Check if the index already exists
            cursor.execute("PRAGMA index_list('emails')")
            indexes = cursor.fetchall()
            if 'idx_email_index' not in [index[1] for index in indexes]:
                # Create index if it doesn't exist
                log.info("Creating index on email_index...")
                cursor.execute("CREATE INDEX idx_email_index ON emails (email_index)")
                conn.commit()
                log.info("Index created successfully.")
            else:
                log.info("Index already exists.")

    def update_categories(self, clustered_emails, cluster_descriptions):
        with self.connect_db() as conn:
            self.ensure_index()
            cursor = conn.cursor()
            for cluster_id, emails in clustered_emails.items():
                category = cluster_descriptions[cluster_id]
                for email in track(emails):
                    email_index = email['Email Index']
                    cursor.execute("UPDATE emails SET category = ? WHERE email_index = ?", (category, email_index))
            conn.commit()

    def cluster(self, num_clusters=10):
        self.ensure_column()

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
                clustered_emails[cluster_id].append({'Email Index': idx, 'Email Info': email_info})

        # Categorize Clusters
        clustered_emails, cluster_descriptions = self.generate_category_descriptions(clustered_emails)
        self.display_clusters(clustered_emails, cluster_descriptions)

        # Update Database
        log.info('Updating database with catagories')
        self.update_categories(clustered_emails, cluster_descriptions)

        return clustered_emails, cluster_descriptions

    def fetch_emails_between(self, email1, email2):
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM emails
                WHERE (from_email = ? AND to_emails LIKE ?)
                OR (to_emails LIKE ? AND from_email = ?)
                """, (email1, f'%{email2}%', email1, f'%{email2}%'))
            columns = [column[0] for column in cursor.description]
            emails = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return emails
    
    def chart_email_categories(self):
        # Connect to the database and fetch categories and their count
        with self.connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, COUNT(*) FROM emails GROUP BY category ORDER BY COUNT(*) DESC")
            category_data = cursor.fetchall()

        # Check if data is available
        if category_data:
            # Unpacking categories and their counts
            categories, counts = zip(*category_data)

            # Creating a bar chart
            plt.figure(figsize=(12, 7))
            plt.barh(categories, counts, color='purple')  # Use horizontal bar chart for better label readability
            plt.xlabel('Number of Emails')
            plt.ylabel('Email Category')
            plt.title('Number of Emails by Category')
            plt.tight_layout()
            plt.savefig('Emails by Catagory.png')
        else:
            log.error("No category data available to plot.")



log = logging.getLogger("rich")
if __name__ == "__main__":
    database = Database('index.faiss', 'emails.db')
    # cluster = database.cluster(num_clusters=50)
    database.chart_email_categories()

