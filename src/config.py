import os

from dotenv import load_dotenv


load_dotenv()


LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

DISTANCE_THRESHOLD = float(
    os.getenv("DISTANCE_THRESHOLD", "1.1")
)

QUERY_REWRITE_COUNT = int(
    os.getenv("QUERY_REWRITE_COUNT", "3")
)