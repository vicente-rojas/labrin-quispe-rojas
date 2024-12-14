# import json
# from src.loaders.loader_pro import DocumentIngestionPipeline
# from src.loaders.pre_process import PreprocesamientoJson
# from src.chunking.semantic_chunking import SemanticChunker
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter 
# from src.retrievers.re_ranker import DocumentReRanker
# from src.retrievers.repacking import DocumentRepacker
# from src.Query_Rewriting.HyDE import HyDE

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

# # Imprimir los primeros 5 chunks para revisar
# # for i, chunk in enumerate(all_chunks[:5]):
# #     print(f"Chunk {i + 1}: {chunk['text']}")
# #     print(f"Metadata: {chunk['metadata']}\n")

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
#     distance=Distance.COSINE,
#     overwrite=True) # Cambio añadiendo el parametro overwrite

# # Preparar textos y metadatos
# metadata = [chunk["metadata"] for chunk in loaded_chunks]

# # Subir los embeddings y metadatos a Qdrant
# metadata = [chunk["metadata"] for chunk in loaded_chunks]
# qdrant_store.add_embeddings(chunk_texts, chunk_embeddings, metadata, batch_size=500)

# Paso 6: Crear una instancia de Retriever con Hybrid Search


# Integrar Document Re-ranking en el pipeline
# Cargar el corpus de textos desde el archivo all_chunks.json
# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Crear una instancia de QueryRewriter utilizando T5-small
# rewriter = QueryRewriter(model_name="t5-small")

# # Crear una instancia de Retriever con el rewriter
# retriever = Retriever(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     collection_name="semantic_chunks",
#     vector_size=768,
#     distance_metric=Distance.COSINE,
#     alpha=0.5,
#     chunk_texts=chunk_texts,
#     query_rewriter=rewriter
# )

# # Ajustar el índice sparse con el corpus de chunks
# retriever.fit_sparse_index(chunk_texts)

# # Crear una instancia de DocumentReRanker
# reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# # Realizar una consulta con Query Rewriting y Hybrid Search
# query = "Why the phenylalanine containing aspartame be declared, and how should it be labeled?"
# rewritten_query = rewriter.rewrite(query)
# print(f"Consulta reformulada: {rewritten_query}")

# # Buscar documentos relevantes con Hybrid Search
# results = retriever.search(rewritten_query, top_k=15, method="hybrid")

# # Mostrar los resultados iniciales
# print("Resultados iniciales:")
# for res in results:
#     print(f"Texto: {res[0]}, Score combinado: {res[1]}")

# # Crear una instancia de HyDE
# hyde = HyDE(generator_model=t5_model, embedding_model=your_embedding_model)

# # Realizar búsqueda con HyDE
# results_with_hyde = hyde.search_with_hyde(query, retriever)

# # Continuar con Document Repacking y Re-ranking
# repacked_results = repacker.repack(query, results_with_hyde)
# reranked_results = reranker.rerank(query, repacked_results)

# # Mostrar los resultados reordenados
# print("\nResultados reordenados (Document Re-ranking):")
# for res in reranked_results:
#     print(f"Texto: {res[0]}, Score re-rankeado: {res[1]}")


#############################################################################

# import json
# from src.loaders.loader_pro import DocumentIngestionPipeline
# from src.loaders.pre_process import PreprocesamientoJson
# from src.chunking.semantic_chunking import SemanticChunker
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.re_ranker import DocumentReRanker
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.HyDE import HyDE
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Inicialización de modelos necesarios
# try:
#     # Modelo T5 para HyDE
#     t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
#     t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
#     generator_model = {"tokenizer": t5_tokenizer, "model": t5_model}

#     # Modelo de embeddings densos
#     embedding_model = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# except Exception as e:
#     print(f"Error al inicializar modelos: {e}")
#     raise

# # Integrar Document Re-ranking en el pipeline
# # Cargar el corpus de textos desde el archivo all_chunks.json
# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# try:
#     retriever = Retriever(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         collection_name="semantic_chunks",
#         vector_size=768,
#         distance_metric=Distance.COSINE,
#         alpha=0.5,
#         chunk_texts=chunk_texts,
#     )

#     retriever.fit_sparse_index(chunk_texts)
#     reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     hyde = HyDE(generator_model=generator_model, embedding_model=embedding_model)

#     # Consulta original sin reformulación
#     #query = "What is the process for obtaining approval for new food additives?"
#     #query = "What are the labeling requirements for food packaging in the United States?"
#     #query = "What regulations apply to the use of preservatives in packaged foods?"
#     #query = "What standards must be met for food additives to be considered safe?"
#     #query = "What regulations govern the labeling of low-acid canned foods?"
#     #query = "How must food products with irradiation processes be labeled?"
#     #query = "What is the regulatory process for setting tolerances for pesticide residues in food?"
#     #query = "What are the CFR requirements for food labeling to avoid consumer deception?"
#     #query = "What are the guidelines for labeling organic food products?" No funciona si el organic esta con ''
#     #query = "What information must be included in the ingredient list of a food product?"
#     #query = " How are low-acid canned foods regulated under food safety laws?"
#     #query = "What standards must be met for food additives to be considered safe?"
#     #query = "What are the standards for declaring food expiration dates on labels?"
#     #query = "What are the requirements for traceability in the food supply chain?"
#     #query = "How is food safety monitored during the transportation of fresh produce?"
#     query = "How are dietary supplements labeled differently from conventional food products?"

#     print(f"Consulta utilizada: {query}")

#     # Recuperación y procesamiento
#     results = retriever.search(query, top_k=15, method="hybrid")
#     results_with_hyde = hyde.search_with_hyde(query, retriever)
#     #print(results_with_hyde)

#     repacker = DocumentRepacker(embedding_model=embedding_model)
#     repacked_results = repacker.repack(query, results_with_hyde)
#     reranked_results = reranker.rerank(query, repacked_results)

#     print("\nResultados reordenados (Document Re-ranking):")
#     for res in reranked_results:
#         print(f"Texto: {res[0]}, Score re-rankeado: {res[1]}")
# except Exception as e:
#     print(f"Error durante la recuperación o reordenamiento: {e}")
#     raise

#############################################

# import json
# from src.loaders.loader_pro import DocumentIngestionPipeline
# from src.chunking.semantic_chunking import SemanticChunker
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.re_ranker import DocumentReRanker
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter
# from src.retrievers.HyDE import HyDE

# # Integrar Document Re-ranking en el pipeline
# # Cargar el corpus de textos desde el archivo all_chunks.json
# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Paso 5: Inicialización de Retriever, Query Rewriter y HyDE
# print("=== Paso 5: Inicialización de herramientas RAG ===")
# query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
# hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
# retriever = Retriever(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     collection_name="semantic_chunks",
#     vector_size=768,
#     distance_metric=Distance.COSINE,
#     alpha=0.5,
#     chunk_texts=chunk_texts
# )

# retriever.fit_sparse_index(chunk_texts)
# reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
# repacker = DocumentRepacker(embedding_model=embedding_generator)

# # Paso 6: Consulta
# print("=== Paso 6: Consulta ===")
# query = "What is the process for obtaining approval for new food additives?"
# print(f"Consulta original: {query}")

# # Reformulación de la consulta
# rewritten_query = query_rewriter.rewrite(query)
# print(f"Consulta reformulada: {rewritten_query}")

# # Recuperación con HyDE
# results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# # Reempacado y re-rankeado
# repacked_results = repacker.repack(rewritten_query, results_with_hyde)
# reranked_results = reranker.rerank(rewritten_query, repacked_results)

# # Mostrar resultados finales
# print("\nResultados reordenados (Document Re-ranking):")
# for res in reranked_results:
#     print(f"Texto: {res[0]}, Score: {res[1]}")


########################### version con ollama ####################################


# import json
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.re_ranker import DocumentReRanker
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter
# from src.retrievers.HyDE import HyDE

# # Integrar Document Re-ranking en el pipeline
# # Cargar el corpus de textos desde el archivo all_chunks.json
# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Paso 5: Inicialización de Retriever, Query Rewriter y HyDE
# print("=== Paso 5: Inicialización de herramientas RAG ===")
# query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
# hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
# retriever = Retriever(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     collection_name="semantic_chunks",
#     vector_size=768,
#     distance_metric=Distance.COSINE,
#     alpha=0.5,
#     chunk_texts=chunk_texts
# )
# retriever.fit_sparse_index(chunk_texts)
# reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
# repacker = DocumentRepacker(embedding_model=embedding_generator)

# # # Paso 6: Consulta
# # print("=== Paso 6: Consulta ===")
# # query = "What is the process for obtaining approval for new food additives?"
# # print(f"Consulta original: {query}")

# # # Reformulación de la consulta
# # rewritten_query = query_rewriter.rewrite(query)
# # print(f"Consulta reformulada: {rewritten_query}")

# # # Recuperación con HyDE
# # results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# # Paso 6: Consulta
# print("=== Paso 6: Consulta ===")
# query ="What standards must be met for food additives to be considered safe?"
# print(f"Consulta original: {query}")

# # Reformulación de la consulta
# rewritten_query = query_rewriter.rewrite(query)
# if not rewritten_query.strip():
#     print("Failed to generate a rewritten query. Using the original query.")
#     rewritten_query = query
# print(f"Consulta reformulada: {rewritten_query}")

# # # Recuperación con HyDE
# # results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# # Generar documento hipotético
# hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
# print("\nDocumento Hipotético Generado por HyDE:\n")
# print(hypothetical_document)

# # Recuperación con HyDE
# results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# # Reempacado y re-rankeado
# repacked_results = repacker.repack(rewritten_query, results_with_hyde)
# reranked_results = reranker.rerank(rewritten_query, repacked_results)

# # Mostrar resultados finales
# print("\nResultados reordenados (Document Re-ranking):")
# for res in reranked_results:
#     print(f"Texto: {res[0]}, Score: {res[1]}")

# #Respuesta del RAG

# # Combinar fragmentos para generar un resumen con Ollama
# combined_text = " ".join([res[0] for res in reranked_results[:5]])  # Limitar a los 5 mejores fragmentos

# # Crear un prompt para Ollama
# prompt = f"""
# Using the following extracted text, generate a coherent paragraph summarizing the process for obtaining approval for new food additives:

# {combined_text}
# """
# import requests

# # Enviar el prompt a Ollama
# payload = {
#     "model": "llama3.2",
#     "messages": [{"role": "user", "content": prompt}]
# }

# try:
#     response = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)
#     if response.status_code == 200:
#         generated_paragraph = ""
#         for line in response.iter_lines(decode_unicode=True):
#             try:
#                 response_json = json.loads(line)
#                 generated_paragraph += response_json.get("message", {}).get("content", "")
#             except json.JSONDecodeError:
#                 continue
#         print("\nGenerated Paragraph:\n", generated_paragraph.strip())
#     else:
#         print(f"Error in Ollama API: {response.status_code} - {response.text}")
# except requests.RequestException as e:
#     print(f"Connection error with Ollama: {e}")



################ Version con  Summarizer.py ############################

import json
from src.embedding.dense_embeddings import DenseEmbeddings
from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
from qdrant_client.http.models import Distance
from src.retrievers.retriever import Retriever
from src.retrievers.repacking import DocumentRepacker
from src.reranker.re_ranker import DocumentReRanker
from src.Query_Rewriting.Query_Rewriting import QueryRewriter
from src.retrievers.HyDE import HyDE
from src.summarizer.Summarizer import OllamaSummarizer

# Cargar el corpus de textos desde el archivo all_chunks.json
with open("all_chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)
chunk_texts = [chunk["text"] for chunk in loaded_chunks]

#Embbeding Generator
embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Inicialización de Retriever, Query Rewriter y HyDE
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

# Retriever y re-rankeador
retriever.fit_sparse_index(chunk_texts)
reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
repacker = DocumentRepacker(embedding_model=embedding_generator)

# Paso Consulta
print("=== Consulta ===")
query ="What standards must be met for food additives to be considered safe?"
print(f"Consulta original: {query}")

# Reformulación de la consulta
rewritten_query = query_rewriter.rewrite(query)
if not rewritten_query.strip():
    print("Failed to generate a rewritten query. Using the original query.")
    rewritten_query = query
print(f"Consulta reformulada: {rewritten_query}")

# Recuperación con HyDE
results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# Generar documento hipotético
hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
print("\nDocumento Hipotético Generado por HyDE:\n")
print(hypothetical_document)

# Recuperación con HyDE
results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)

# Reempacado y re-rankeado
repacked_results = repacker.repack(rewritten_query, results_with_hyde)
reranked_results = reranker.rerank(rewritten_query, repacked_results)

# Mostrar resultados finales
print("\nResultados reordenados (Document Re-ranking):")
for res in reranked_results:
    print(f"Texto: {res[0]}, Score: {res[1]}")

# Generar resumen con OllamaSummarizer

# Inicializar OllamaSummarizer
api_url = "http://localhost:11434/api/chat"
ollama_summarizer = OllamaSummarizer(api_url=api_url, model="llama3.2")

# Define un contexto dinámico basado en el resultado
dynamic_context = "Summary of the relevant chunks and key findings for the query"

# Generar resumen con un contexto personalizado
try:
    summary = ollama_summarizer.generate_summary(
        reranked_results=reranked_results, 
        num_fragments=5, 
        context=dynamic_context
    )
    print("\nGenerated Summary:\n", summary)
except Exception as e:
    print(f"An error occurred: {e}")
