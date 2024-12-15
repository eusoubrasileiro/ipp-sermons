from aiohttp import web
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.utils import ComponentDevice
from haystack.components.rankers import TransformersSimilarityRanker
from haystack import Pipeline
from utils import load_documents_and_embeddings, config 
from pathlib import Path 
import argparse

def loadHaystakPipeline(root_data_path : Path) -> Pipeline:
    # Load documents and embeddings from disk
    document_store = load_documents_and_embeddings(root_data_path)
    text_embedder = SentenceTransformersTextEmbedder(
        model=config['rag']['models']['sentence-transformer'], device=ComponentDevice.from_str("cuda:0")
    )
    embedding_retriever = InMemoryEmbeddingRetriever(document_store)
    embedding_retriever.top_k = 30  # Adjust semantic results
    bm25_retriever = InMemoryBM25Retriever(document_store)
    bm25_retriever.top_k = 30  # Adjust keyword results
    # the first retriever is the bm25 the second is the embedding retriever
    document_joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion") # Adjust weights
    # already better than default
    #join_mode="distribution_based_rank_fusion" == reciprocal_rank_fusion == 
    #document_joiner = DocumentJoiner(weights=[0.3, 0.7]) # Adjust weights
    ranker = TransformersSimilarityRanker(model=config['rag']['models']['reranker-transformer'])
    ranker.top_k = 30

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

    return pipeline


async def handle_suggestion(request):
    data = await request.json()
    suggestion = data.get("suggestion", "")
    if suggestion.strip():  # Verificar se a sugestão não está vazia
        with open("sugestoes.txt", "a", encoding="utf-8") as file:
            file.write(suggestion + "\n")
        return web.json_response({"status": "success", "message": "Sugestão salva com sucesso!"})
    return web.json_response({"status": "error", "message": "Sugestão vazia."}, status=400)


# Define query endpoint
async def query(request):
    data = await request.json()
    query_text = data.get("query", "")
    
    if not query_text:
        return web.json_response({"error": "No query provided"}, status=400)
    
    # Run the query
    pipeline = request.app['pipeline']
    result = pipeline.run({"text_embedder": {"text": query_text}, "bm25_retriever": {"query": query_text}, "ranker": {"query": query_text}})

    # Format results
    response = {
        "query": query_text,
        "results": [
            {
                "title": doc.meta.get("name"),
                "content": doc.content,  # Preview all text
                "score": doc.score,
                "spot_url" : doc.meta.get("spotify_url") if isinstance(doc.meta.get("spotify_url"), str) else None,
                "sdc_url" : doc.meta.get("soundcloud_url") if isinstance(doc.meta.get("soundcloud_url"), str) else None,
                "duration" : doc.meta.get('duration_str'),
                "sent_min" : f"{doc.meta.get('sentences_min'):.2f} sentences/minute",
                "artist" : doc.meta.get("artist") if isinstance(doc.meta.get("artist"), str) else None                
            }
            for doc in result["ranker"]["documents"]
        ]
    }
    return web.json_response(response)

# Serve frontend
async def index(request):
    return web.FileResponse((request.app['index_path'] / "index.html").absolute())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Query for Sermons WebServer')
    parser.add_argument('-d','--data-path', help='Root folder with embeddings/ and docs/ metadata for haystack semantic query', required=True)    
    parser.add_argument('-i','--index-path', help='Folder for static data (index.html)', required=True)    
    args = parser.parse_args()
    # Setup routes
    app = web.Application()
    app.router.add_post("/query", query)
    app.router.add_get("/", index)
    app.router.add_post("/suggestion", handle_suggestion)
    app['index_path'] = Path(args.index_path).absolute()
    app['pipeline'] = loadHaystakPipeline(Path(args.data_path))
    web.run_app(app, host="0.0.0.0", port=8890)
