"""
File: weaviate_client.py
Description: Class and helper functions related to Weaviate functions.
Author: Matthew Stefanovic
Email: matthew@stefanovic.us
Created: 2024-07-19

Changelog:
    Version 1.0.0 - 2024-07-14 - Matthew Stefanovic
        - Added the __init__(), connect(), and close() function to the Weaviate_Client class
        - Tested functions to make sure they work
"""

"""
Methods this class needs: 
    - [ ] Connect and disconnect to a client
    - [ ] Create a collection based on a given schema
    - [ ] Check if a collection has been created
    - [ ] Batch adds entries to a collection
"""
import weaviate
import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")


class Weaviate_Client:
    def __init__(self):
        self.client = None

    def connect(self):
        """Connects to the Weaviate client"""
        log.info(f'Trying to connect to Weaviate client')
        self.client = weaviate.connect_to_local()
        log.info(f'Connected to Weaviate client')

    def close(self):
        """Closes the connection to the Weiviate client"""
        log.info(f'Closing Weaviate client connection')
        self.client.close()


if __name__ == '__main__':
    weaviate_client = Weaviate_Client()
    weaviate_client.connect()
    weaviate_client.close()
