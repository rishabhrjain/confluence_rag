from pathlib import Path

CONFLUENCE_BASE_URL = "https://<add your domain>.atlassian.net"
USERNAME = "<CONFLUENCE USERNAME>" 
API_TOKEN = "CONFLUENCE API TOKEN"

# Dirs
BASE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = BASE_DIR / "cache"
SPACES_DIR = CACHE_DIR / "spaces"
PAGES_DIR = CACHE_DIR / "pages"
CONTENTS_DIR = CACHE_DIR / "page_content"

RAG_DIR = CACHE_DIR / "rag"
CHUNKS_DIR = RAG_DIR / "chunks"

EVAL_DIR = RAG_DIR / "rag_eval"


# AWS bedrock configs
AWS_PROFILE = ''
AWS_REGION = ''

# LLM model id
MODEL_ID = "arn:aws:bedrock:<AWS_REGION>:<ID>:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"


# Embedding model id
EMBEDDING_MODEL_ID = "cohere.embed-english-v3"




