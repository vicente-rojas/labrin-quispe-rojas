# import streamlit as st
# from src.retrievers.retriever import Retriever 

# # Configuración Inicial de la app
# st.set_page_config(page_title="Búsqueda Semántica", layout="wide")

# # Crear una instancia de la clase Retriever
# retriever = Retriever(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     collection_name="semantic_chunks",
#     vector_size=768,
#     distance_metric="COSINE")

# # Título de la aplicación
# st.title("Búsqueda Semántica con Qdrant")

# # Entrada de la consulta del usuario
# query = st.text_input("Introduce tu consulta:", placeholder="Ejemplo: Why must the presence of phenylalanine in drugs containing aspartame be declared, and how should it be labeled?/What is the definition of artificial flavor?")

# # Botón para realizar la búsqueda
# if st.button("Buscar"):
#     if query.strip():
#         # Usar el método `search` de la clase Retriever
#         results = retriever.search(query, top_k=5)

#         # Mostrar resultados
#         st.subheader("Resultados de Búsqueda")
#         if results:
#             for i, res in enumerate(results):
#                 st.markdown(f"### Resultado {i + 1}")
#                 st.write(f"**Score:** {res['score']}")
#                 st.write(f"**Texto:** {res['payload'].get('text', 'Sin texto disponible')}")
#                 st.write(f"**Metadata:** {res['payload'].get('metadata', 'Sin metadatos disponibles')}")
#                 st.markdown("---")
#         else:
#             st.warning("No se encontraron resultados.")
#     else:
#         st.warning("Por favor, introduce una consulta válida.")

#########################Primera Versiona############################################

# import streamlit as st
# import json
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.re_ranker import DocumentReRanker
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter
# from src.retrievers.HyDE import HyDE
# from src.summarizer.Summarizer import OllamaSummarizer

# # Configuración inicial de Streamlit
# st.title("RAG Query Application")
# st.markdown("Perform queries and retrieve summaries using a RAG pipeline.")

# # Cargar el corpus de textos desde el archivo
# @st.cache_data
# def load_chunks(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# loaded_chunks = load_chunks("all_chunks.json")
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Inicialización de herramientas RAG
# @st.cache_resource
# def initialize_tools():
#     embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     retriever = Retriever(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         collection_name="semantic_chunks",
#         vector_size=768,
#         distance_metric=Distance.COSINE,
#         alpha=0.5,
#         chunk_texts=chunk_texts
#     )
#     retriever.fit_sparse_index(chunk_texts)
#     reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     repacker = DocumentRepacker(embedding_model=embedding_generator)
#     query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
#     hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
#     summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")
#     return embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer

# embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer = initialize_tools()

# # Entrada del usuario
# query = st.text_input("Enter your query:", "What standards must be met for food additives to be considered safe?")
# if st.button("Run Query"):
#     # Paso 1: Reformulación de la consulta
#     rewritten_query = query_rewriter.rewrite(query)
#     st.write(f"Rewritten Query: {rewritten_query}")

#     # Paso 2: Recuperación con HyDE
#     with st.spinner("Retrieving documents..."):
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)
    
#     # Paso 3: Generar documento hipotético
#     hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
#     st.subheader("Hypothetical Document Generated by HyDE:")
#     st.write(hypothetical_document)

#     # Paso 4: Reempacado y re-rankeo
#     repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#     reranked_results = reranker.rerank(rewritten_query, repacked_results)

#     # Mostrar resultados reordenados
#     st.subheader("Reranked Results:")
#     for res in reranked_results:
#         st.write(f"**Text:** {res[0]}")
#         st.write(f"**Score:** {res[1]}")

#     # Paso 5: Generar resumen
#     with st.spinner("Generating summary..."):
#         dynamic_context = "Summary of the relevant chunks and key findings for the query"
#         try:
#             summary = summarizer.generate_summary(
#                 reranked_results=reranked_results, 
#                 num_fragments=5, 
#                 context=dynamic_context
#             )
#             st.subheader("Generated Summary:")
#             st.write(summary)
#         except Exception as e:
#             st.error(f"An error occurred while generating the summary: {e}")

######################Segunda Version###############################################

# import streamlit as st
# import json
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.re_ranker import DocumentReRanker
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter
# from src.retrievers.HyDE import HyDE
# from src.summarizer.Summarizer import OllamaSummarizer

# # Configuración inicial de Streamlit
# st.title("RAG Query Application")
# st.markdown("Perform queries and retrieve summaries using a RAG pipeline.")

# # Cargar el corpus de textos desde el archivo
# @st.cache_data
# def load_chunks(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# loaded_chunks = load_chunks("all_chunks.json")
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Inicialización de herramientas RAG
# @st.cache_resource
# def initialize_tools():
#     embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     retriever = Retriever(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         collection_name="semantic_chunks",
#         vector_size=768,
#         distance_metric=Distance.COSINE,
#         alpha=0.5,
#         chunk_texts=chunk_texts
#     )
#     retriever.fit_sparse_index(chunk_texts)
#     reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     repacker = DocumentRepacker(embedding_model=embedding_generator)
#     query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
#     hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
#     summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")
#     return embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer

# embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer = initialize_tools()

# # Entrada del usuario
# query = st.text_input("Enter your query:", "What standards must be met for food additives to be considered safe?")
# if st.button("Run Query"):
#     # Paso 1: Reformulación de la consulta
#     rewritten_query = query_rewriter.rewrite(query)
#     st.write(f"Rewritten Query: {rewritten_query}")

#     # Paso 2: Recuperación con HyDE
#     with st.spinner("Retrieving documents..."):
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)
    
#     # Paso 3: Generar documento hipotético
#     hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
#     st.subheader("Hypothetical Document Generated by HyDE:")
#     st.write(hypothetical_document)

#     # Paso 4: Reempacado y re-rankeo
#     repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#     reranked_results = reranker.rerank(rewritten_query, repacked_results)

#     # Mostrar resultados reordenados
#     st.subheader("Reranked Results:")
#     for res in reranked_results:
#         st.write(f"**Text:** {res[0]}")
#         st.write(f"**Score:** {res[1]}")

#     # Paso 5: Generar resumen
#     with st.spinner("Generating summary..."):
#         dynamic_context = "Summary of the relevant chunks and key findings for the query"
#         try:
#             summary = summarizer.generate_summary(
#                 reranked_results=reranked_results, 
#                 num_fragments=5, 
#                 context=dynamic_context
#             )
#             st.subheader("Generated Summary:")
#             st.write(summary)
#         except Exception as e:
#             st.error(f"An error occurred while generating the summary: {e}")

######################Tercera Version###############################################

# import streamlit as st
# import json
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.re_ranker import DocumentReRanker
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter
# from src.retrievers.HyDE import HyDE
# from src.summarizer.Summarizer import OllamaSummarizer

# # Configuración inicial de Streamlit
# st.title("RAG Query Application")
# st.markdown("Perform queries and retrieve summaries using a RAG pipeline.")

# # Cargar el corpus de textos desde el archivo
# @st.cache_data
# def load_chunks(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# loaded_chunks = load_chunks("all_chunks.json")
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Inicialización de herramientas RAG
# @st.cache_resource
# def initialize_tools():
#     embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     retriever = Retriever(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         collection_name="semantic_chunks",
#         vector_size=768,
#         distance_metric=Distance.COSINE,
#         alpha=0.5,
#         chunk_texts=chunk_texts
#     )
#     retriever.fit_sparse_index(chunk_texts)
#     reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     repacker = DocumentRepacker(embedding_model=embedding_generator)
#     query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
#     hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
#     summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")
#     return embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer

# embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer = initialize_tools()

# # Entrada del usuario
# query = st.text_input("Enter your query:", "What standards must be met for food additives to be considered safe?")
# if st.button("Run Query"):
#     # Paso 1: Reformulación de la consulta
#     rewritten_query = query_rewriter.rewrite(query)
#     st.write(f"Rewritten Query: {rewritten_query}")

#     # Paso 2: Recuperación con HyDE
#     with st.spinner("Retrieving documents..."):
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)
#         # Extraer similitudes de los resultados
#         similarities = []
#         if isinstance(results_with_hyde, list):
#             for result in results_with_hyde:
#                 if isinstance(result, tuple) and len(result) > 1:
#                     similarities.append(result[1])  # Asumiendo que la similitud está en el índice 1
#                 else:
#                     similarities.append(None)  # Manejar casos inesperados
#         else:
#             st.error("Unexpected structure in results_with_hyde.")
#             similarities = []

#     # Paso 3: Generar documento hipotético
#     hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
#     st.subheader("Hypothetical Document Generated by HyDE:")
#     st.write(hypothetical_document)

#     # Paso 4: Reempacado y re-rankeo
#     repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#     reranked_results = reranker.rerank(rewritten_query, repacked_results)

#     # Mostrar resultados reordenados con similitudes
#     st.subheader("Reranked Results with Similarities:")
#     for idx, (res, sim) in enumerate(zip(reranked_results, similarities)):
#         st.write(f"**Result {idx + 1}:**")
#         st.write(f"**Text:** {res[0]}")
#         st.write(f"**Score:** {res[1]}")
#         st.write(f"**Similarity:** {sim}")

#     # Paso 5: Generar resumen
#     with st.spinner("Generating summary..."):
#         dynamic_context = "Summary of the relevant chunks and key findings for the query"
#         try:
#             summary = summarizer.generate_summary(
#                 reranked_results=reranked_results, 
#                 num_fragments=5, 
#                 context=dynamic_context
#             )
#             st.subheader("Generated Summary:")
#             st.write(summary)
#         except Exception as e:
#             st.error(f"An error occurred while generating the summary: {e}")


######################Cuarta Version###############################################

import streamlit as st
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

# Configuración inicial de Streamlit
st.title("RAG Query Application")
st.markdown("Perform queries and retrieve summaries using a RAG pipeline.")

# Cargar el corpus de textos desde el archivo
@st.cache_data
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Cargar los chunks antes de inicializar herramientas
loaded_chunks = load_chunks("all_chunks.json")
chunk_texts = [chunk["text"] for chunk in loaded_chunks]  # Definición de chunk_texts

# Inicialización de herramientas RAG
@st.cache_resource
def initialize_tools():
    embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    retriever = Retriever(
        model_name="sentence-transformers/all-mpnet-base-v2",
        collection_name="semantic_chunks",
        vector_size=768,
        distance_metric=Distance.COSINE,
        alpha=0.5,
        chunk_texts=chunk_texts  # Ahora chunk_texts está definido
    )
    retriever.fit_sparse_index(chunk_texts)
    reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    repacker = DocumentRepacker(embedding_model=embedding_generator)
    query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
    hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
    summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")
    return embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer

embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer = initialize_tools()

# Lista de posibles consultas predefinidas
possible_queries = [
    "What standards must be met for food additives to be considered safe?",
    "How are dietary supplements labeled differently from conventional food products?",
    "What are the labeling requirements for organic food products?",
    "What is the regulatory process for approving new preservatives?",
    "How must food packaging be labeled for allergens?"
]

# Selector de consulta predefinida
selected_query = st.selectbox("Select a predefined query to test:", options=possible_queries)

# Entrada del usuario con consulta predefinida seleccionada
query = st.text_input("Enter your query:", selected_query)

# Botón para ejecutar la consulta
if st.button("Run Query"):
    # Paso 1: Reformulación de la consulta
    rewritten_query = query_rewriter.rewrite(query)
    st.write(f"Rewritten Query: {rewritten_query}")

    # Paso 2: Recuperación con HyDE
    with st.spinner("Retrieving documents..."):
        results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)
        # Extraer similitudes de los resultados
        similarities = []
        if isinstance(results_with_hyde, list):
            for result in results_with_hyde:
                if isinstance(result, tuple) and len(result) > 1:
                    similarities.append(result[1])  # Asumiendo que la similitud está en el índice 1
                else:
                    similarities.append(None)  # Manejar casos inesperados
        else:
            st.error("Unexpected structure in results_with_hyde.")
            similarities = []

    # Paso 3: Generar documento hipotético
    hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
    st.subheader("Hypothetical Document Generated by HyDE:")
    st.write(hypothetical_document)

    # Paso 4: Reempacado y re-rankeo
    repacked_results = repacker.repack(rewritten_query, results_with_hyde)
    reranked_results = reranker.rerank(rewritten_query, repacked_results)

    # Mostrar resultados reordenados con similitudes
    st.subheader("Reranked Results with Similarities:")
    for idx, (res, sim) in enumerate(zip(reranked_results, similarities)):
        st.write(f"**Result {idx + 1}:**")
        st.write(f"**Text:** {res[0]}")
        st.write(f"**Score:** {res[1]}")
        st.write(f"**Similarity:** {sim}")

    # Paso 5: Generar resumen
    with st.spinner("Generating summary..."):
        dynamic_context = "Summary of the relevant chunks and key findings for the query"
        try:
            summary = summarizer.generate_summary(
                reranked_results=reranked_results, 
                num_fragments=5, 
                context=dynamic_context
            )
            st.subheader("Generated Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred while generating the summary: {e}")



