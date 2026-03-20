import logging
from backend.jobs import JobManager
from backend.neo4j_utils import Neo4jManager
from backend.reranker import DocumentReranker
from backend.config import (
    WORKING_DIR,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)

# Setup base logger so routers can use it
base_logger = logging.getLogger("uvicorn.error")

# Instantiate managers
job_manager = JobManager(WORKING_DIR)

neo4j_manager = Neo4jManager(
    uri=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

document_reranker = DocumentReranker()


# Create a state container for things that load asynchronously during lifespan
class AppState:
    def __init__(self):
        self.rag = None


state = AppState()
