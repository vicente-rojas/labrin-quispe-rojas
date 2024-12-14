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

# Consultas predefinidas para pruebas
possible_queries = [
    "What standards must be met for food additives to be considered safe?",
    "How are dietary supplements labeled differently from conventional food products?",
    "What are the labeling requirements for organic food products?",
    "What is the regulatory process for approving new preservatives?",
    "How must food packaging be labeled for allergens?",
    "What is the process for obtaining approval for new food additives?",
    "What are the labeling requirements for food packaging in the United States?",
    "What regulations apply to the use of preservatives in packaged foods?",
    "What standards must be met for food additives to be considered safe?",
    "What regulations govern the labeling of low-acid canned foods?",
    "How must food products with irradiation processes be labeled?",
    "What is the regulatory process for setting tolerances for pesticide residues in food?",
    "What are the CFR requirements for food labeling to avoid consumer deception?",
    "What are the guidelines for labeling organic food products?",
    "What information must be included in the ingredient list of a food product?",
    "How are low-acid canned foods regulated under food safety laws?",
    "What standards must be met for food additives to be considered safe?",
    "What are the standards for declaring food expiration dates on labels?",
    "What are the requirements for traceability in the food supply chain?",
    "How is food safety monitored during the transportation of fresh produce?",
    "How are dietary supplements labeled differently from conventional food products?"
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

    # Recuperación con HyDE
    with st.spinner("Retrieving documents..."):
        results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)
        # Extraer similitudes de los resultados
        similarities = []
        if isinstance(results_with_hyde, list):
            for result in results_with_hyde:
                if isinstance(result, tuple) and len(result) > 1:
                    similarities.append(result[1])  
                else:
                    similarities.append(None)  
        else:
            st.error("Unexpected structure in results_with_hyde.")
            similarities = []

    # Generar documento hipotético
    hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
    st.subheader("Hypothetical Document Generated by HyDE:")
    st.write(hypothetical_document)

    # Repack y rerank
    repacked_results = repacker.repack(rewritten_query, results_with_hyde)
    reranked_results = reranker.rerank(rewritten_query, repacked_results)

    # Mostrar resultados reordenados con similitudes
    st.subheader("Reranked Results with Similarities:")
    for idx, (res, sim) in enumerate(zip(reranked_results, similarities)):
        st.write(f"**Result {idx + 1}:**")
        st.write(f"**Text:** {res[0]}")
        st.write(f"**Score:** {res[1]}")
        st.write(f"**Similarity:** {sim}")

    # Generar resumen
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




