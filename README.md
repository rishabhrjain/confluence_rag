# confluence_rag
A LLM powered RAG application to improve search results on Confluence.

Here are the high level steps 

## Initial setup
1. Install poetry 
2. Run `poety install`
3. Run `poetry shell` to activate the virtual env

## Data collection

1. Data is fetched through confluence APIs in a atlas doc format which is confluence specific format
2. The atlas doc is parsed to extract all relevant page content in addition to metadata like page_url, title, etc. This metadata is very helpful for adding more context to chunks and the response (adding clickable page_urls so users can easily visit source pages). There are a lot of other functionalities that can be explored like date of creation/modification. This way we can give priority to more latest information during retrieval. 

This retrieves data from all public/team spaces in confluence. This can be expanded to pull pages from personal confluence space as well. 

Refer to `confluence_rag/src/fetch_confluence_data.py` for code. 

## Chunking, Indexing and Retrieval
We use Llamaindex as our orchestration layer for our RAG sysem. This framework is utilized for chunking, indexing and retrieval

You can easily modify chunking parameters, embedding model, vector databases and similarity metric. Currently, the vector indices are just sored as .pkl files but these will be moved to vector database soon. 

Retrieves top-k nodes using semantic similarity. A hybrid retrieval strategy can be used as well.

 You can set all these parameters when initializing the ConfluenceRAGWithLlamaIndex in `confluence_rag/src/confluence_rag_llamaindex.py` 

## Prompt Engineering

Do not ignore this step. This is a low hanging fruit to improve our LLM's response. Do not jump into complex techincal things before optimizing this one. 
I use custom propmpt template from Llamaindex. 

## Response Generation
I use AWS Bedrock models for response generation. You can easily modify this in `confluence_rag/src/confluence_rag_llamaindex.py` to point to a different LLM. 

## Evaluation 
I think this is the most important part. As for every ML system, without evaluation it would be hard to set a baseline and quantify improvements in performance. This helps us to easily choose between different techniques

For now, i have evaluation for the retriever which is the main component that would determine the overall accuracy of the system. For response generation usingLLM model , i am using standard models on AWS which already have performance benchmarks hence I don't have specific evaluation. For custom LLM's it would be nice to have some evaluation benchmarks.

### Evaluation strategy
I use a binary relevance strategy meaning given a query i check whether the relevant document (confluence page) is fetched in the top-k results.
Designing the strategy this way helped me to easily leverage LLMs to generate evaluation data which i could then manually review. 

### How was evaluation data generated? 

1. Randomly select 100 confluence pages from the dataset (this makes sure there is diversity in the eval dataset)
2. For each page, ask the LLM to generate 5 queries of varied difficulty relevant to this document. 
3. Manually review the queries to make sure they are relevant to the document. 

Now we can use these queries on our retriever and check if the relevant page is retrieved in top-k results. This way we can calculate metrics like recall@top-k. 

This evaluation strategy is not ideal but still is pretty useful with a very little human effort. Ideally we would have more refined strategy based on resources. 

checkout `confluence_rag/src/evaluation` for more details.


### STREAMLIT UI
Finally there is a streamlit UI which can be deployed to make it easier to query the RAG system. 
Just run `streamlit run confluence_rag/src/app.py` to start a server. 




