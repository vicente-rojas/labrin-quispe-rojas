# Aplicacion V1.0: Primera entrega
# Funcionalidades:  Consulta y recuperación de los chunks relevantes a la vector srtore.
# 
# import streamlit as st
# import json
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.retrievers.repacking import DocumentRepacker
# from src.reranker.re_ranker import DocumentReRanker
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

# # Cargar los chunks antes de inicializar herramientas
# loaded_chunks = load_chunks("all_chunks.json")
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]  # Definición de chunk_texts

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
#         chunk_texts=chunk_texts  # Ahora chunk_texts está definido
#     )
#     retriever.fit_sparse_index(chunk_texts)
#     reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
#     repacker = DocumentRepacker(embedding_model=embedding_generator)
#     query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
#     hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
#     summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")
#     return embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer

# embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer = initialize_tools()

# # Consultas predefinidas para pruebas
# possible_queries = [
#     "What standards must be met for food additives to be considered safe?",
#     "How are dietary supplements labeled differently from conventional food products?",
#     "What are the labeling requirements for organic food products?",
#     "What is the regulatory process for approving new preservatives?",
#     "How must food packaging be labeled for allergens?",
#     "What is the process for obtaining approval for new food additives?",
#     "What are the labeling requirements for food packaging in the United States?",
#     "What regulations apply to the use of preservatives in packaged foods?",
#     "What standards must be met for food additives to be considered safe?",
#     "What regulations govern the labeling of low-acid canned foods?",
#     "How must food products with irradiation processes be labeled?",
#     "What is the regulatory process for setting tolerances for pesticide residues in food?",
#     "What are the CFR requirements for food labeling to avoid consumer deception?",
#     "What are the guidelines for labeling organic food products?",
#     "What information must be included in the ingredient list of a food product?",
#     "How are low-acid canned foods regulated under food safety laws?",
#     "What standards must be met for food additives to be considered safe?",
#     "What are the standards for declaring food expiration dates on labels?",
#     "What are the requirements for traceability in the food supply chain?",
#     "How is food safety monitored during the transportation of fresh produce?",
#     "How are dietary supplements labeled differently from conventional food products?"
# ]

# # Selector de consulta predefinida
# selected_query = st.selectbox("Select a predefined query to test:", options=possible_queries)

# # Entrada del usuario con consulta predefinida seleccionada
# query = st.text_input("Enter your query:", selected_query)

# # Botón para ejecutar la consulta
# if st.button("Run Query"):
#     # Paso 1: Reformulación de la consulta
#     rewritten_query = query_rewriter.rewrite(query)
#     st.write(f"Rewritten Query: {rewritten_query}")

#     # Recuperación con HyDE
#     with st.spinner("Retrieving documents..."):
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=15)
#         # Extraer similitudes de los resultados
#         similarities = []
#         if isinstance(results_with_hyde, list):
#             for result in results_with_hyde:
#                 if isinstance(result, tuple) and len(result) > 1:
#                     similarities.append(result[1])  
#                 else:
#                     similarities.append(None)  
#         else:
#             st.error("Unexpected structure in results_with_hyde.")
#             similarities = []

#     # Generar documento hipotético
#     hypothetical_document = hyde.generate_hypothetical_document(rewritten_query)
#     st.subheader("Hypothetical Document Generated by HyDE:")
#     st.write(hypothetical_document)

#     # Repack y rerank
#     repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#     reranked_results = reranker.rerank(rewritten_query, repacked_results)

#     # Mostrar resultados reordenados con similitudes
#     st.subheader("Reranked Results with Similarities:")
#     for idx, (res, sim) in enumerate(zip(reranked_results, similarities)):
#         st.write(f"**Result {idx + 1}:**")
#         st.write(f"**Text:** {res[0]}")
#         st.write(f"**Score:** {res[1]}")
#         st.write(f"**Similarity:** {sim}")

#     # Generar resumen
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

##############################################################################################################
# Aplicacion V2.0: Segunda entrega (Entrega Final)

import streamlit as st
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding.dense_embeddings import DenseEmbeddings
from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
from qdrant_client.http.models import Distance
from src.retrievers.retriever import Retriever
from src.retrievers.repacking import DocumentRepacker
from src.reranker.re_ranker import DocumentReRanker
from src.Query_Rewriting.Query_Rewriting import QueryRewriter
from src.retrievers.HyDE import HyDE
from src.summarizer.Summarizer import OllamaSummarizer

# Configuración inicial de la aplicacion de Streamlit
st.title("INF3822 Proyecto Aplicado I - Retrieval Augmented Generation (RAG)")
st.markdown("Realiza consultas y rescata informacion usando un RAG pipeline.")

# Inicializar historial en session_state
if "history" not in st.session_state:
    st.session_state.history = []

# Cargar el corpus de textos desde el archivo
@st.cache_data
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

loaded_chunks = load_chunks("all_chunks.json")
chunk_texts = [chunk["text"] for chunk in loaded_chunks]  

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
        chunk_texts=chunk_texts)
    
    try:
        retriever.fit_sparse_index(chunk_texts)
    except Exception as e:
        st.warning("Collection 'semantic_chunks' already exists. Skipping initialization.")

    reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    repacker = DocumentRepacker(embedding_model=embedding_generator)
    query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
    hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)
    summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")
    return embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer

embedding_generator, retriever, reranker, repacker, query_rewriter, hyde, summarizer = initialize_tools()

# Cargar archivo de Ground Truth
#Archivo de preguntas y respuestas generado con gpt4o y almacenado en un archivo JSON. 
#Por temas de tiempo y recursos (calidad de las preguntas y ground thruth) no se logró integrar
# dentro del pipeline la generacion automatica y se opto por cargarlo directamente.

@st.cache_data
def load_ground_truth(file_path="ground_truth.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

ground_truth = load_ground_truth()

# Se muestran las preguntas disponibles en el sidebar
st.sidebar.title("Preguntas")
questions = [q["question"] for q in ground_truth["questions"]]
selected_question = st.sidebar.selectbox("Selecciona una pregunta", questions)

# Se obtienen los datos de la pregunta seleccionada
question_data = next(item for item in ground_truth["questions"] if item["question"] == selected_question)
st.subheader(f"Pregunta: {selected_question}")
st.write(f"Respuesta esperada: {question_data['ground_truth']}")

# Evaluación al presionar el botón
if st.button("Evaluar"):
    st.write("Ejecutando evaluación...")

    # Reformulación de la consulta (Query Rewriting)
    rewritten_query = query_rewriter.rewrite(selected_question)
    st.write(f"**Consulta reformulada:** {rewritten_query}")

    # Recuperación básica (naive retrieval)
    with st.spinner("Ejecutando recuperación básica..."):
        basic_results = retriever.search(rewritten_query, top_k=3)
        #st.write("Resultados brutos obtenidos del retriever:")
        #st.json(basic_results)  # Depuración de la estructura

        # Ajuste para manejar lista de tuplas
        if basic_results and isinstance(basic_results, list) and len(basic_results) > 0:
            first_result = basic_results[0]
            
            # Si el primer resultado es una tupla, extraer el texto
            if isinstance(first_result, tuple) and len(first_result) > 0:
                basic_context_text = first_result[0]  
            elif isinstance(first_result, str):  
                basic_context_text = first_result
            else:
                basic_context_text = "El formato del resultado no contiene texto válido."
                st.warning("No se pudo identificar el formato esperado en el resultado.")
        else:
            basic_context_text = "No hay resultados."
            st.warning("La lista de resultados está vacía o no es válida.")

        #st.write(f"**Texto del contexto básico:** {basic_context_text}")

    # Recuperación avanzada (advanced retrieval)
    with st.spinner("Ejecutando recuperación avanzada..."):
        results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=5)
        advanced_context_text = results_with_hyde[0]["payload"]["text"] if results_with_hyde else "No hay resultados."
        repacked_results = repacker.repack(rewritten_query, results_with_hyde)
        advanced_results = reranker.rerank(rewritten_query, repacked_results)
        advanced_summary = summarizer.generate_summary(reranked_results=advanced_results, num_fragments=5)

    # Tabla comparativa
    st.subheader("Resultados de la Consulta")
    comparison_data = {
        "Consulta": [rewritten_query],
        "Resultado Básico": [basic_context_text],
        "Resumen Avanzado": [advanced_summary]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

    # Calcular métricas de evaluación
    # No se pudo implementar directamente Ragas por limitaciones de tokens de OpenIA.
    # Se optó por utilizar la similitud coseno entre los embeddings de las respuestas esperadas y las generadas.

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    def calculate_metrics(context_text, summary, query, true_answer, embedding_generator):
        # Verificar entradas válidas
        if not context_text or not summary or not query or not true_answer:
            return {"error": "Input texts cannot be empty"}
        
        # Generar embeddings y normalizar
        try:
            summary_emb = normalize(embedding_generator.embed_documents([summary]))
            true_answer_emb = normalize(embedding_generator.embed_documents([true_answer]))
            query_emb = normalize(embedding_generator.embed_documents([query]))
            context_emb = normalize(embedding_generator.embed_documents([context_text]))
        except Exception as e:
            return {"error": f"Embedding generation failed: {e}"}

        # Calcular métricas
        faithfulness = cosine_similarity(summary_emb, true_answer_emb)[0][0]
        answer_relevancy = cosine_similarity(summary_emb, query_emb)[0][0]
        precision = cosine_similarity(context_emb, true_answer_emb)[0][0]
        recall = cosine_similarity(true_answer_emb, context_emb)[0][0]
        #f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Faithfulness": faithfulness,
            "Answer Relevancy": answer_relevancy,
            "Context Precision": precision,
            "Context Recall": recall,
            #"Context F1-Score": f1_score
        }

    # Comparar contexto básico con la respuesta esperada
    basic_metrics = calculate_metrics(basic_context_text, basic_context_text, selected_question, question_data["ground_truth"], embedding_generator=embedding_generator)
    # Comparar resumen avanzado con la respuesta esperada
    advanced_metrics = calculate_metrics(advanced_context_text, advanced_summary, selected_question, 
                                         question_data["ground_truth"], embedding_generator=embedding_generator)

    # Definir las métricas esperadas
    expected_metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]

    # Asegurarse de que los valores correspondan a las métricas esperadas
    basic_values = [basic_metrics.get(metric, 0) for metric in expected_metrics]
    advanced_values = [advanced_metrics.get(metric, 0) for metric in expected_metrics]

    # Crear el DataFrame de métricas
    metrics_df = pd.DataFrame({
        "Métrica": expected_metrics,
        "Recuperación Básica": basic_values,
        "Recuperación Avanzada": advanced_values
    })

    # Mostrar la tabla en Streamlit
    st.subheader("Métricas de Evaluación")
    st.table(metrics_df)

    # Guardar resultados en el historial
    st.session_state.history.append({
        "question": selected_question,
        "rewritten_query": rewritten_query,
        "basic_context": basic_context_text,
        "advanced_context": advanced_context_text,
        "metrics": {
            "basic": basic_metrics,
            "advanced": advanced_metrics
        }
    })

    # # Mostrar métricas
    # st.subheader("Métricas de Evaluación")
    # metrics_df = pd.DataFrame({
    #     "Métrica": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
    #     "Recuperación Básica": list(basic_metrics.values()),
    #     "Recuperación Avanzada": list(advanced_metrics.values())
    # })

    # st.table(metrics_df)

    # Gráfico de barras comparativo
    melted_metrics = metrics_df.melt(id_vars="Métrica", var_name="Método", value_name="Puntuación")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Métrica", y="Puntuación", hue="Método", data=melted_metrics)
    plt.title("Comparación de Métricas: Recuperación Básica vs Avanzada")
    plt.ylabel("Puntuación")
    plt.ylim(0, 1)
    st.pyplot(plt)

    #print("Resultados Básicos:", basic_context_text)
    #print("Resultados Avanzados:", advanced_context_text)

# Mostrar historial
st.sidebar.subheader("Historial de Evaluaciones")
if st.session_state.history:
    for idx, entry in enumerate(reversed(st.session_state.history)):
        st.sidebar.write(f"### Evaluación {len(st.session_state.history) - idx}")
        st.sidebar.write(f"**Pregunta:** {entry['question']}")
        st.sidebar.write(f"**Consulta reformulada:** {entry['rewritten_query']}")
        st.sidebar.write(f"**Texto básico:** {entry['basic_context']}")
        st.sidebar.write(f"**Texto avanzado:** {entry['advanced_context']}")
        st.sidebar.write(f"**Métricas:** {entry['metrics']}")
        st.sidebar.write("---")
else:
    st.sidebar.write("No se han realizado evaluaciones aún.")




