from aiohttp import web
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.utils import ComponentDevice
from haystack.components.rankers import TransformersSimilarityRanker
from haystack import Pipeline
from utils import load_documents_and_embeddings

# Load documents and embeddings from disk
document_store = load_documents_and_embeddings()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # faster: "sentence-transformers/all-MiniLM-L6-v2"

text_embedder = SentenceTransformersTextEmbedder(
    model=MODEL_NAME, device=ComponentDevice.from_str("cuda:0")
)
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
embedding_retriever.top_k = 40  # Adjust semantic results
bm25_retriever = InMemoryBM25Retriever(document_store)
bm25_retriever.top_k = 20  # Adjust keyword results
# the first retriever is the bm25 the second is the embedding retriever
document_joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion") # Adjust weights
# already better than default
#join_mode="distribution_based_rank_fusion" == reciprocal_rank_fusion == 
#document_joiner = DocumentJoiner(weights=[0.3, 0.7]) # Adjust weights
ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

pipeline = Pipeline()
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("embedding_retriever", embedding_retriever)
pipeline.add_component("bm25_retriever", bm25_retriever)
pipeline.add_component("document_joiner", document_joiner)
pipeline.add_component("ranker", ranker)
pipeline.connect("text_embedder", "embedding_retriever")
pipeline.connect("bm25_retriever", "document_joiner")
pipeline.connect("embedding_retriever", "document_joiner")
pipeline.connect("document_joiner", "ranker")

# Define query endpoint
async def query(request):
    data = await request.json()
    query_text = data.get("query", "")
    
    if not query_text:
        return web.json_response({"error": "No query provided"}, status=400)
    
    # Run the query
    result = pipeline.run({"text_embedder": {"text": query_text}, "bm25_retriever": {"query": query_text}, "ranker": {"query": query_text}})
    
    # Format results
    response = {
        "query": query_text,
        "results": [
            {
                "title": doc.meta.get("title"),
                "content": doc.content,  # Preview first 300 characters
                "score": doc.score
            }
            for doc in result["ranker"]["documents"]
        ]
    }
    return web.json_response(response)

# Serve frontend
async def index(request):
    return web.FileResponse("index.html")

# Setup routes
app = web.Application()
app.router.add_post("/query", query)
app.router.add_get("/", index)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8890)
