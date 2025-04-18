import json
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import boto3

# LlamaIndex imports
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter, NodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.embeddings import BaseEmbedding
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
# from llama_index.core.retrievers.bm25_retriever import BM25Retriever
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.llms.bedrock import Bedrock
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage

from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from confluence_rag.config import CHUNKS_DIR, PAGES_DIR, AWS_PROFILE, MODEL_ID, EMBEDDING_MODEL_ID, AWS_REGION


class ConfluenceRAGWithLlamaIndex:
    def __init__(
        self, 
        data_path: str = None, 
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        top_k = 3, 
        embedding_model: Optional[BaseEmbedding] = None,
        llm_model: Optional[Any] = None, 
        use_llm: Optional[bool] = True
    ):
        """
        Initialize the RAG system for Confluence data using LlamaIndex.
        
        Args:
            data_path: Path to the JSON file containing Confluence data
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            embedding_model: LlamaIndex embedding model (default: OpenAIEmbedding)
            llm_model: Optional LLM for generation
        """
        self.data_path = data_path
        self.pages = []
        self.nodes = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

         # AWS settings
        session = boto3.Session(profile_name=AWS_PROFILE)
        self.bedrock_client = session.client(
                            service_name='bedrock-runtime',
                            region_name=AWS_REGION  
                        )
        
        # Set up embedding model
        self.embedding_model = embedding_model or HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # REPLACE THE SIMPLE EMBEDDING MODEL WITH AWS OR ANY EMBEDDING MODEL
        # self.embedding_model = BedrockEmbedding(
        #         client=self.bedrock_client,
        #         model_name=EMBEDDING_MODEL_ID, 
        #         aws_region=AWS_REGION
        #     )
        Settings.embed_model = self.embedding_model
        
        
        # Node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )

        self.top_k = top_k
        
       
        
        if use_llm:
            if not llm_model:
                llm_model = Bedrock(
                                client=self.bedrock_client,
                                model=MODEL_ID,  
                                context_size=4000, 

                                # model generation params
                                temperature=0.1, # Lower values (0.0-0.3): More deterministic, focused responses
                                max_tokens=2000, 
                                top_p = 0.1  # Lower values (0.1-0.3): Very focused on high-probability tokens
                                
                            )
            self.llm_model = llm_model
        else:
            self.llm_model = None
        
        
        Settings.llm = self.llm_model
        
        
    def load_data(self) -> None:
        """Load the Confluence data from JSON file."""
        if self.data_path.exists():
            with open(self.data_path, 'r') as f:
                self.pages = json.load(f)
            print(f"Loaded {len(self.pages)} pages from {self.data_path}")
        else:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def process_data(self) -> List[BaseNode]:
        """
        Process the data and create nodes (chunks) for each page.
        
        Returns:
            List of processed nodes
        """
        all_documents = []

        # Load data
        self.load_data()
        
        # Convert Confluence pages to LlamaIndex documents
        for page in self.pages:
            page_content = page.get('page_content', '')
            
            # Create document with metadata
            doc = Document(
                text=page.get("page_title") + ": \n" + page_content, # adding page title
                metadata={
                    "page_id": page.get("page_id"),
                    "page_title": page.get("page_title"),
                    "page_url": page.get("page_url"),
                    "space_id": page.get("space_id"),
                    "created_at": page.get("created_at"),
                }
            )
            all_documents.append(doc)
        
        # Parse documents into nodes (chunks)
        self.nodes = self.node_parser.get_nodes_from_documents(all_documents)
        
        print(f"Created {len(self.nodes)} nodes from {len(self.pages)} pages")
        
        return self.nodes
    
    def create_index(self, nodes: Optional[List[BaseNode]] = None) -> VectorStoreIndex:
        """
        Create a vector store index from the nodes.
        
        Args:
            nodes: Optional list of nodes (if None, uses self.nodes)
            
        Returns:
            VectorStoreIndex
        """
        if nodes is None:
            if not self.nodes:
                self.process_data()
            nodes = self.nodes
        
        # Create vector store index
        index = VectorStoreIndex(nodes)
        
        return index
    
    def save_nodes_and_index(self, 
                           index: VectorStoreIndex,
                           nodes_path: str, 
                           index_dir: str) -> None:
        """
        Save nodes and index to disk.
        
        Args:
            index: VectorStoreIndex to save
            nodes_path: Path to save nodes
            index_dir: Dir to save Vector index
        """
        # Save nodes
        with open(nodes_path, "wb") as f:
            pickle.dump(self.nodes, f)
            print(f"Saved {len(self.nodes)} nodes to {nodes_path}")
        
        # Save index
        index.storage_context.persist(persist_dir=index_dir)
        print(f"Saved index to {index_dir}")
    
    def load_index(self, index_dir: str) -> VectorStoreIndex:
        """
        Load index from disk.
        
        Args:
            index_path: Path to load index from
            
        Returns:
            Loaded VectorStoreIndex
        """
        
        
        # Load index
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        
        print(f"Loaded index from {index_dir}")
        
        return index
    
    def load_nodes(self, nodes_path: str) -> List[BaseNode]:
        """
        Load nodes from disk.
        
        Args:
            nodes_path: Path to load nodes from
            
        Returns:
            List of nodes
        """
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file {nodes_path} not found")
        
        # Load nodes
        with open(nodes_path, "rb") as f:
            self.nodes = pickle.load(f)
            
        print(f"Loaded {len(self.nodes)} nodes from {nodes_path}")
        
        return self.nodes
    
    def search(self, query: str, index: Optional[VectorStoreIndex] = None, 
              ) -> List[Dict[str, Any]]:
        """
        Search for the most relevant nodes given a query.
        
        Args:
            query: Search query
            index: Optional VectorStoreIndex (if None, creates a new one)
            top_k: Number of top results to return
            
        Returns:
            List of top k node dictionaries with similarity scores
        """
        if index is None:
            index = self.create_index()
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k,
        )

        retrieved_nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for i, node in enumerate(retrieved_nodes):
            results.append({
                "node_id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
                "similarity": node.score if hasattr(node, 'score') else None,
                "rank": i + 1
            })
        
        return retrieved_nodes, results
    


    def generate_answer(self, query: str, index: Optional[VectorStoreIndex] = None) -> str:
        """
        Generate an answer based on retrieved nodes with AWS Bedrock LLM.
        
        Args:
            query: User query
            index: Optional VectorStoreIndex (if None, creates a new one)
            
        Returns:
            Generated answer
        """
        if index is None:
            index = self.create_index()

        nodes, _ = self.search(query, index)
        
        if not nodes:
            return "No relevant information found."
        
        source_urls = set([node.metadata.get('page_url') for node in nodes])
        
        final_response = 'Source Pages:'+ "\n\n".join(source_urls)

        # Custom prompt template
        custom_prompt = PromptTemplate(
            """You are a helpful AI assistant tasked with answering questions based on the provided context.
            
            Context information is below:
            ---------------------
            {context_str}
            ---------------------
            
            Given this information, please answer the following question thoroughly and accurately. List any links or resources mentioned in context. 
            If the answer cannot be determined from the context, state that clearly.
            
            Question: {query_str}
            
            Answer:"""
        )
        
        if not self.llm_model:
            # If no LLM is provided, just return the top retrieved nodes
            context = "\n\n".join([node.text for node in nodes])
            return f"Based on the most relevant information found:\n\n{context}"
        
        # Create a response synthesizer with the custom prompt
        response_synthesizer = get_response_synthesizer(
            llm=self.llm_model,
            text_qa_template=custom_prompt
        )
        
        # Generate response using the custom synthesizer and retrieved nodes
        response = response_synthesizer.synthesize(
            query=query,
            nodes=nodes # the top k nodes are passed here
        )
        
        final_response += "\n\n" + str(response)
        return final_response


def main():
    
    data_path = PAGES_DIR / "cleaned_pages.json"
    # nodes_path = CHUNKS_DIR / 'chunk_nodes_cohere_embedding.pkl'
    # index_dir = CHUNKS_DIR / 'chunk_index_cohere_embedding'
    nodes_path = CHUNKS_DIR / 'chunk_nodes.pkl'
    index_dir = CHUNKS_DIR / 'chunk_index'

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    
    
    # Initialize RAG system
    rag = ConfluenceRAGWithLlamaIndex(
        data_path=data_path,
        chunk_size=512,
        chunk_overlap=128,
        top_k=5,
        use_llm=True
    )
    
    if nodes_path.exists() and index_dir.exists():
        print("Loading pre-computed nodes and index...")
        
        # Load nodes and index
        nodes = rag.load_nodes(nodes_path)
        index = rag.load_index(index_dir)

    else:
    
        print("Computing and storing nodes and index...")
        
        # Process data and create nodes
        nodes = rag.process_data()
        
        # Create index
        index = rag.create_index(nodes)
        
        # Save nodes and index
        rag.save_nodes_and_index(index, nodes_path, index_dir)
    
    # Search example
    search_query = "I am part of ML team. Provide me all the onboarding docs"
    
    answer = rag.generate_answer(search_query, index)
    print("LLM generated answer: \n", answer)

if __name__ == "__main__":
    main()