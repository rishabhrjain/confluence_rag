
import json 
import numpy as np
import json
from tqdm import tqdm

from confluence_rag.config import PAGES_DIR, EVAL_DIR, CHUNKS_DIR
from confluence_rag.src.confluence_rag_llamaindex import ConfluenceRAGWithLlamaIndex


data_path = PAGES_DIR / "cleaned_pages.json"
nodes_path = CHUNKS_DIR / 'chunk_nodes.pkl'
index_path = CHUNKS_DIR / 'chunk_index.pkl'


# Initialize RAG system for evaluating retriever
rag = ConfluenceRAGWithLlamaIndex(
    top_k=5,
    use_llm=False
)



if nodes_path.exists() and index_path.exists():
    print("Loading pre-computed nodes and index...")
    
    # Load nodes and index
    nodes = rag.load_nodes(nodes_path)
    index = rag.load_index(index_path)

else:
    raise ValueError("Nodes and Index not found")


# Read eval data and search the index
with open(EVAL_DIR / 'rag_eval.json', 'rb') as f:
    eval_data = json.load(f)

recall_count= 0
total_q = 0
i = 1
for eval in tqdm(eval_data, total=len(eval_data)): 

    eval_page_id = eval['page_id']
    questions = eval['questions']
    total_q += len(questions)

    for question in questions:
        _, retreived_results = rag.search(question, index)

        retrieved_page_ids = [node['metadata'].get('page_id') for node in retreived_results]

        if eval_page_id in retrieved_page_ids:
            recall_count += 1

    recall_top_k = (recall_count) / (total_q)
    print("Recall@top3: ", recall_top_k)
    i += 1


        
             

