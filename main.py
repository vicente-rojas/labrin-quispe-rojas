# #Solo pdf's

# from src.loaders.loader import PDFIngestionPipeline

# #Paso 1: Carga de PDF desde Carpeta

# pdf_loader = PDFIngestionPipeline("data")
# pdf_loader.load_pipeline()

import json
from src.loaders.loader_pro import DocumentIngestionPipeline
from src.loaders.pre_process import PreprocesamientoJson
from src.chunking.semantic_chunking import SemanticChunker
from src.embedding.dense_embeddings import DenseEmbeddings
from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
from qdrant_client.http.models import Distance

# # Paso 1: Carga de PDF desde Carpeta

# pdf_loader = DocumentIngestionPipeline("data-2")
# pdf_loader.load_pipeline()

# # Paso 2: Preprocesamiento del json

# preprocessor = PreprocesamientoJson("documents_text.json", "cleaned_docs.json")
# preprocessor.clean()

# # Paso 3: Chunking (Semantic Chunking)

# cos_dist_limit = 0.15
# all_chunks = []

# with open("cleaned_docs.json", "r", encoding="utf-8") as f:
#     cleaned_docs_ = json.load(f)

# # Procesa cada documento en `cleaned_docs_`
# for doc_index, doc in enumerate(cleaned_docs_):
#     text = doc["text"]

#     # Inicializar el SemanticChunker
#     semantic_chunker = SemanticChunker(document=text, buffer_size=1, model_name="sentence-transformers/all-mpnet-base-v2")
#     chunks = semantic_chunker.split_into_chunks(threshold=cos_dist_limit)
    
#     # Add metadata to each chunk
#     for chunk_text in chunks:
#         chunk_data = {
#             "text": chunk_text,
#             "metadata": {
#                 "document_index": doc_index,
#                 "original_document_length": len(text)
#             }
#         }
#         all_chunks.append(chunk_data)

# # Imprime los primeros 5 chunks para revisar
# for i, chunk in enumerate(all_chunks[:5]):
#     print(f"Chunk {i + 1}: {chunk['text']}")
#     print(f"Metadata: {chunk['metadata']}\n")

# # Guardar `all_chunks` en un archivo json
# with open("all_chunks.json", "w", encoding="utf-8") as f:
#     json.dump(all_chunks, f, ensure_ascii=False, indent=4)

# #Paso 4: Generacion de Embeddings   

# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)

# # Inicializar DenseEmbeddings
# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Generar embeddings para los textos de los chunks
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]
# chunk_embeddings = embedding_generator.embed_documents(chunk_texts)

# # Paso 5: qdrant vector Store

# #Inicializar QdrantVectorStore
# qdrant_store = QdrantVectorStore(
#     embedding_model=embedding_generator,
#     collection_name="semantic_chunks",
#     vector_size=768,  
#     distance=Distance.COSINE)

# # Preparar textos y metadatos
# metadata = [chunk["metadata"] for chunk in loaded_chunks]

# # Subir los embeddings y metadatos a Qdrant
# metadata = [chunk["metadata"] for chunk in loaded_chunks]
# qdrant_store.add_embeddings(chunk_texts, chunk_embeddings, metadata, batch_size=500)

# # Tiempo paso 4 y 5 ~ 5 min

# Paso 6: Consulta Sobre el Vector Data Base

# Inicializar DenseEmbeddings para consultas
embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Inicializar QdrantVectorStore con la colección existente
qdrant_store = QdrantVectorStore(
    embedding_model=embedding_generator,
    collection_name="semantic_chunks",  
    vector_size=768,
    distance=Distance.COSINE)

# Realizar una consulta sobre la base de datos vectorial
query = "como se muestra el etiquetado en un envase"
results = qdrant_store.search(query, top_k=5)

# Mostrar los resultados
print("Resultados de búsqueda:")
for res in results:
    print(f"Score: {res['score']}, Metadata: {res['payload']}")
