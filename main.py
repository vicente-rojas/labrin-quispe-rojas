# Description: Este script es el punto de entrada principal para ejecutar el flujo de trabajo de RAG.
# Author: Grupo 5 - Proyecto Aplicado I
# Date: 2024-12-20
# Entrega N°2 - Entrega Final

# Paso 0 : Carga de librearias necesarias

import json
from src.loaders.loader_pro import DocumentIngestionPipeline
from src.loaders.pre_process import PreprocesamientoJson
from src.chunking.semantic_chunking import SemanticChunker
from src.embedding.dense_embeddings import DenseEmbeddings
from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
from qdrant_client.http.models import Distance
from src.retrievers.retriever import Retriever
from src.Query_Rewriting.Query_Rewriting import QueryRewriter 
from src.reranker.re_ranker import DocumentReRanker
from src.retrievers.repacking import DocumentRepacker
from src.retrievers.HyDE import HyDE
from src.summarizer.Summarizer import OllamaSummarizer

#####################################  Etapa 1: Chunks, Embeddings y Vector Store ########################################  

# '''Si ya se genero el VD no es necesario ejecutar la etapa 1, del paso 1 al 5 y se 
# puede saltar a la etapa 2'''

# # Paso 1: Carga de Informacion (PDF | .docx | .txt) desde la carpeta data-2
# pdf_loader = DocumentIngestionPipeline("data-2")
# pdf_loader.load_pipeline()

# # Paso 2: Preprocesamiento del texto (json)
# preprocessor = PreprocesamientoJson("documents_text.json", "cleaned_docs.json")
# preprocessor.clean()

# # Paso 3: Chunking del texto (Semantic Chunking - distacia coseno)
# cos_dist_limit = 0.15
# all_chunks = []

# with open("cleaned_docs.json", "r", encoding="utf-8") as f:
#     cleaned_docs_ = json.load(f)

# # Se procesa cada documento en `cleaned_docs_`
# for doc_index, doc in enumerate(cleaned_docs_):
#     text = doc["text"]

#     # Se inicializa la instancia del SemanticChunker
#     semantic_chunker = SemanticChunker(document=text, buffer_size=1, model_name="sentence-transformers/all-mpnet-base-v2")
#     chunks = semantic_chunker.split_into_chunks(threshold=cos_dist_limit)
    
#     # Se añaden los chunks a la lista `all_chunks`
#     '''Se añade algunos parametos como metadatos para cada chunk, como el indice del documento original y la longitud del documento original.
#     Sin embargo, la cantidad de metadatos de los documentos no es suficiente como para plantear una solucion de busqueda eficiente que los utilice,
#     como por ejemplo: self-query retirever'''

#     for chunk_text in chunks:
#         chunk_data = {
#             "text": chunk_text,
#             "metadata": {
#                 "document_index": doc_index,
#                 "original_document_length": len(text)
#             }
#         }
#         all_chunks.append(chunk_data)

# # Chequeo de la etapa 3: Imprimir los primeros 5 chunks.
# # for i, chunk in enumerate(all_chunks[:5]):
# #     print(f"Chunk {i + 1}: {chunk['text']}")
# #     print(f"Metadata: {chunk['metadata']}\n")

# # Guardar `all_chunks` en un archivo json
# '''Esta etapa es importante para guardar los chunks generados en un archivo json, ya que estos chunks se utilizaran para generar los embeddings.
# Se define como etapa de intermedia para poder reutilizar los chunks generados en caso de que se necesite generar nuevamente los embeddings.'''
# with open("all_chunks.json", "w", encoding="utf-8") as f:
#     json.dump(all_chunks, f, ensure_ascii=False, indent=4)

# # Paso 4: Generacion de Embeddings (Dense Embeddings)   
# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)

# # Inicializar DenseEmbeddings
# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Generar embeddings para los textos de los chunks
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]
# chunk_embeddings = embedding_generator.embed_documents(chunk_texts)

# # Paso 5: QdrantVectorStore (Base de datos de vectores)

# #Inicializar QdrantVectorStore
# qdrant_store = QdrantVectorStore(
#     embedding_model=embedding_generator,
#     collection_name="semantic_chunks",
#     vector_size=768,  
#     distance=Distance.COSINE,
#     overwrite=True) # Si es True, se sobreescribirán los vectores existentes en la colección (ojo)

# # Preparar textos y metadatos
# metadata = [chunk["metadata"] for chunk in loaded_chunks]

# # Subir los embeddings y metadatos a Qdrant
# metadata = [chunk["metadata"] for chunk in loaded_chunks]
# qdrant_store.add_embeddings(chunk_texts, chunk_embeddings, metadata, batch_size=500)

##########################################  Etapa 2: Advanced Retriever ##########################################  

# Paso 6: Cargar el corpus de textos desde el archivo all_chunks.json

with open("all_chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)
chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# Inicializar el Embbeding Generator
embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Inicializar instancia de Retriever, Query Rewriter y HyDE
print("=== Inicialización de herramientas RAG ===")

query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")

hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)

retriever = Retriever(
    model_name="sentence-transformers/all-mpnet-base-v2",
    collection_name="semantic_chunks",
    vector_size=768,
    distance_metric=Distance.COSINE,
    alpha=0.5,
    chunk_texts=chunk_texts)

retriever.fit_sparse_index(chunk_texts)

reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
repacker = DocumentRepacker(embedding_model=embedding_generator)

# Paso 7: Consulta y recuperación
print("=== Consulta ===")
query ="What standards must be met for food additives to be considered safe?"
print(f"Consulta original: {query}")

# Paso 8: Reformulación de la consulta utilizando LLM (Ollama)
rewritten_query = query_rewriter.rewrite(query)
if not rewritten_query.strip():
    print("Failed to generate a rewritten query. Using the original query.")
    rewritten_query = query
print(f"Consulta reformulada: {rewritten_query}")

# Paso 9: Recuperación con HyDE (Hybrid Document Embeddings - Ollama)
results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# Generación de un documento hipotético con HyDE
hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
print("\nDocumento Hipotético Generado por HyDE:\n")
print(hypothetical_document)

# Recuperación con HyDE
results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# Paso 10: Re-ranking y Re-packing
repacked_results = repacker.repack(rewritten_query, results_with_hyde)
reranked_results = reranker.rerank(rewritten_query, repacked_results)

# Mostrar resultados relevantes en funcion de la consulta
print("\nResultados reordenados (Document Re-ranking):")
for res in reranked_results:
    print(f"Texto: {res[0]}, Score: {res[1]}")

# Paso 11: Generación de resumen con Ollama (funcion ollama_summarizer.generate_summary)

api_url = "http://localhost:11434/api/chat"
ollama_summarizer = OllamaSummarizer(api_url=api_url, model="llama3.2")

# Define un contexto dinámico basado en el resultado
dynamic_context = "Summary of the relevant chunks and key findings for the query"

try:
    summary = ollama_summarizer.generate_summary(
        reranked_results=reranked_results, 
        num_fragments=5, 
        context=dynamic_context
    )
    print("\nGenerated Summary:\n", summary)
except Exception as e:
    print(f"An error occurred: {e}")
