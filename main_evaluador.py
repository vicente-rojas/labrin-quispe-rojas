# import json
# from datasets import Dataset
# from ragas.evaluation import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# import openai

# # Importación de componentes personalizados
# import json
# from dotenv import load_dotenv
# from src.loaders.loader_pro import DocumentIngestionPipeline
# from src.loaders.pre_process import PreprocesamientoJson
# from src.chunking.semantic_chunking import SemanticChunker
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter 
# from src.reranker.re_ranker import DocumentReRanker
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.HyDE import HyDE
# from src.summarizer.Summarizer import OllamaSummarizer

# # Configuración inicial
# load_dotenv()

# # Configuración inicial
# ground_truth_path = "ground_truth.json"
# ollama_url = "http://localhost:11434/api/chat"
# model_name = "llama3.2"

# # Inicializar el Cargador de Documentos
# with open("all_chunks.json", "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Inicializar el Embbeding Generator
# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Inicializar instancia de Retriever, Query Rewriter y HyDE
# print("=== Inicialización de herramientas RAG ===")

# query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")

# hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)

# retriever = Retriever(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     collection_name="semantic_chunks",
#     vector_size=768,
#     distance_metric=Distance.COSINE,
#     alpha=0.5,
#     chunk_texts=chunk_texts)

# retriever.fit_sparse_index(chunk_texts)

# reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
# repacker = DocumentRepacker(embedding_model=embedding_generator)
# ollama_summarizer = OllamaSummarizer(api_url=ollama_url, model="llama3.2")

# def evaluate_rag_pipeline():
#     # Cargar ground truths
#     with open(ground_truth_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Listas para almacenar datos del pipeline
#     questions = []
#     answers = []
#     references = []
#     retrieved_contexts = []

#     for entry in data["questions"]:
#         question = entry["question"]
#         ground_truth = entry["ground_truth"]

#         print(f"\nEvaluando pregunta: {question}\n")

#         # Reformulación de consulta con Llama
#         rewritten_query = query_rewriter.rewrite(question)
#         print(f"Consulta reformulada: {rewritten_query}")

#         # Recuperación con HyDE
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=10)
        
#         # Repacking y Reranking
#         repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#         reranked_results = reranker.rerank(rewritten_query, repacked_results)

#         # Formato de retrieved_contexts
#         formatted_contexts = [str(res[0]) for res in reranked_results]
#         retrieved_contexts.append(formatted_contexts)

#         # Generación de resumen con Llama
#         summary = ollama_summarizer.generate_summary(reranked_results, num_fragments=5, context="")
#         print(f"Resumen generado: {summary}")

#         # Almacenar datos para evaluación
#         questions.append(rewritten_query)
#         answers.append(summary)
#         references.append(ground_truth)

#     # Crear dataset para evaluación
#     dataset_dict = {
#         "question": questions,
#         "answer": answers,
#         "reference": references,
#         "retrieved_contexts": retrieved_contexts,
#     }
#     dataset = Dataset.from_dict(dataset_dict)

#     # Evaluación con ragas usando OpenAI
#     print("\nIniciando evaluación con ragas...\n")
#     result = evaluate(
#         dataset=dataset,
#         llm=openai.Completion.create,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#         ],
#     )

#     # Mostrar resultados
#     print("\nResultados de la Evaluación:")
#     print(result)


# if __name__ == "__main__":
#     evaluate_rag_pipeline()

# import json
# import os
# from datasets import Dataset
# from ragas.evaluation import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# # Importación de componentes personalizados desde el pipeline
# from src.loaders.loader_pro import DocumentIngestionPipeline
# from src.loaders.pre_process import PreprocesamientoJson
# from src.chunking.semantic_chunking import SemanticChunker
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter 
# from src.reranker.re_ranker import DocumentReRanker
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.HyDE import HyDE
# from src.summarizer.Summarizer import OllamaSummarizer

# # Cargar variables de entorno
# load_dotenv('src\.env')

# # Configuración de la API de OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("La clave OPENAI_API_KEY no está configurada correctamente.")

# # Configuración del pipeline y paths
# GROUND_TRUTH_PATH = "ground_truth.json"
# ALL_CHUNKS_PATH = "all_chunks.json"
# OLLAMA_URL = "http://localhost:11434/api/chat"
# MODEL_NAME = "llama3.2"

# # Cargar chunks semánticos
# with open(ALL_CHUNKS_PATH, "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Inicializar componentes del pipeline
# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# query_rewriter = QueryRewriter(ollama_url=OLLAMA_URL, model_name=MODEL_NAME)
# hyde = HyDE(ollama_url=OLLAMA_URL, model_name=MODEL_NAME, embedding_model=embedding_generator)

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
# ollama_summarizer = OllamaSummarizer(api_url=OLLAMA_URL, model=MODEL_NAME)

# # Configurar modelo para RAGAS
# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model="gpt-3.5-turbo",
#     temperature=0,
#     max_tokens=512
# )

# def evaluate_rag_pipeline():
#     """Evalúa el pipeline RAG usando ground truths y métricas de RAGAS."""
#     # Cargar ground truths
#     with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Inicializar listas para evaluación
#     questions, answers, references, retrieved_contexts = [], [], [], []

#     for entry in data["questions"]:
#         question = entry["question"]
#         ground_truth = entry["ground_truth"]

#         print(f"\nEvaluando pregunta: {question}\n")

#         # Reformular la pregunta
#         rewritten_query = query_rewriter.rewrite(question)
#         print(f"Consulta reformulada: {rewritten_query}")

#         # Recuperar documentos relevantes con HyDE
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=10)

#         # Reordenar y empaquetar resultados
#         repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#         reranked_results = reranker.rerank(rewritten_query, repacked_results)

#         # Contextos recuperados
#         formatted_contexts = [res["text"] for res in reranked_results]
#         retrieved_contexts.append(formatted_contexts)

#         # Generar resumen
#         summary = ollama_summarizer.generate_summary(reranked_results, num_fragments=5, context="")
#         print(f"Resumen generado: {summary}")

#         # Almacenar datos para evaluación
#         questions.append(rewritten_query)
#         answers.append(summary)
#         references.append(ground_truth)

#     # Crear dataset para RAGAS
#     dataset_dict = {
#         "question": questions,
#         "answer": answers,
#         "reference": references,
#         "retrieved_contexts": retrieved_contexts,
#     }
#     dataset = Dataset.from_dict(dataset_dict)

#     # Evaluar el pipeline con RAGAS
#     print("\nIniciando evaluación con RAGAS...")
#     result = evaluate(
#         dataset=dataset,
#         llm=llm,
#         metrics=[faithfulness, answer_relevancy, context_precision]
#     )

#     # Mostrar resultados
#     print("\nResultados de la Evaluación:")
#     print(result)

# if __name__ == "__main__":
#     evaluate_rag_pipeline()


# import json
# import os
# from datasets import Dataset
# from ragas.evaluation import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Importación de componentes personalizados desde el pipeline
# from src.loaders.loader_pro import DocumentIngestionPipeline
# from src.loaders.pre_process import PreprocesamientoJson
# from src.chunking.semantic_chunking import SemanticChunker
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance
# from src.retrievers.retriever import Retriever
# from src.Query_Rewriting.Query_Rewriting import QueryRewriter 
# from src.reranker.re_ranker import DocumentReRanker
# from src.retrievers.repacking import DocumentRepacker
# from src.retrievers.HyDE import HyDE
# from src.summarizer.Summarizer import OllamaSummarizer

# # Cargar variables de entorno
# load_dotenv()

# # Configuración del pipeline y paths
# GROUND_TRUTH_PATH = "ground_truth.json"
# ALL_CHUNKS_PATH = "all_chunks.json"
# MODEL_NAME = "gpt2"

# # Inicializar modelo local
# print("Inicializando modelo local...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# # Función para generación de respuestas locales
# def generate_local_response(prompt, max_length=100, temperature=0.7):
#     result = hf_pipeline(prompt, max_length=max_length, temperature=temperature, num_return_sequences=1)
#     return result[0]["generated_text"]

# # Cargar chunks semánticos
# with open(ALL_CHUNKS_PATH, "r", encoding="utf-8") as f:
#     loaded_chunks = json.load(f)
# chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# # Inicializar componentes del pipeline
# embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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
# ollama_summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")

# def evaluate_rag_pipeline():
#     """Evalúa el pipeline RAG usando ground truths y métricas de RAGAS."""
#     # Cargar ground truths
#     with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Inicializar listas para evaluación
#     questions, answers, references, retrieved_contexts = [], [], [], []

#     for entry in data["questions"]:
#         question = entry["question"]
#         ground_truth = entry["ground_truth"]

#         print(f"\nEvaluando pregunta: {question}\n")

#         # Reformular la pregunta
#         rewritten_query = query_rewriter.rewrite(question)
#         print(f"Consulta reformulada: {rewritten_query}")

#         # Recuperar documentos relevantes con HyDE
#         results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=10)

#         # Reordenar y empaquetar resultados
#         repacked_results = repacker.repack(rewritten_query, results_with_hyde)
#         reranked_results = reranker.rerank(rewritten_query, repacked_results)

#         # Contextos recuperados
#         formatted_contexts = [res[0]["text"] if isinstance(res[0], dict) else res[0] for res in reranked_results]
#         retrieved_contexts.append(formatted_contexts)

#         # Generar resumen usando modelo local
#         summary = generate_local_response(prompt=question, max_length=150, temperature=0.7)
#         print(f"Resumen generado: {summary}")

#         # Almacenar datos para evaluación
#         questions.append(rewritten_query)
#         answers.append(summary)
#         references.append(ground_truth)

#     # Crear dataset para evaluación
#     dataset_dict = {
#         "question": questions,
#         "answer": answers,
#         "reference": references,
#         "retrieved_contexts": retrieved_contexts,
#     }
#     dataset = Dataset.from_dict(dataset_dict)

#     # Evaluar el pipeline con RAGAS
#     print("\nIniciando evaluación...")
#     result = evaluate(
#         dataset=dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision]
#     )

#     # Mostrar resultados
#     print("\nResultados de la Evaluación:")
#     print(result)

# if __name__ == "__main__":
#     evaluate_rag_pipeline()

import json
import os
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

# Cargar variables de entorno
load_dotenv()

# Configuración del pipeline y paths
GROUND_TRUTH_PATH = "ground_truth.json"
ALL_CHUNKS_PATH = "all_chunks.json"
MODEL_NAME = "gpt2"

# Detectar dispositivo disponible
device = 0 if torch.cuda.is_available() else -1
print(f"Usando dispositivo: {'GPU' if device == 0 else 'CPU'}")

# Inicializar modelo local
print("Inicializando modelo local...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Función para generación de respuestas locales
def generate_local_response(prompt, max_length=100, temperature=0.7):
    result = hf_pipeline(prompt, max_length=max_length, temperature=temperature, num_return_sequences=1)
    return result[0]["generated_text"]

# Inicializar modelo de embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_embeddings(text):
    """Calcula los embeddings para un texto dado."""
    return embedding_model.encode([text], convert_to_tensor=True)

def cosine_similarity_score(vector1, vector2):
    """Calcula la similitud coseno entre dos vectores."""
    return cosine_similarity(vector1, vector2)[0][0]

def faithfulness(response, context):
    """Mide si la respuesta se basa fielmente en el contexto."""
    response_embedding = calculate_embeddings(response)
    context_embedding = calculate_embeddings(context)
    return cosine_similarity_score(response_embedding, context_embedding)

def answer_relevancy(response, question):
    """Mide la relevancia de la respuesta para la pregunta."""
    response_embedding = calculate_embeddings(response)
    question_embedding = calculate_embeddings(question)
    return cosine_similarity_score(response_embedding, question_embedding)

def context_recall(response, context):
    """Mide cuán exhaustivamente la respuesta utiliza el contexto."""
    response_embedding = calculate_embeddings(response)
    context_embedding = calculate_embeddings(context)
    return cosine_similarity_score(context_embedding, response_embedding)

def context_precision(response, context):
    """Mide cuánta información de la respuesta es relevante para el contexto."""
    response_embedding = calculate_embeddings(response)
    context_embedding = calculate_embeddings(context)
    return cosine_similarity_score(response_embedding, context_embedding)

# Cargar chunks semánticos
with open(ALL_CHUNKS_PATH, "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)
chunk_texts = [chunk["text"] for chunk in loaded_chunks]

# Inicializar componentes del pipeline
embedding_generator = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
query_rewriter = QueryRewriter(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2")
hyde = HyDE(ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=embedding_generator)

retriever = Retriever(
    model_name="sentence-transformers/all-mpnet-base-v2",
    collection_name="semantic_chunks",
    vector_size=768,
    distance_metric=Distance.COSINE,
    alpha=0.5,
    chunk_texts=chunk_texts
)
retriever.fit_sparse_index(chunk_texts)

reranker = DocumentReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
repacker = DocumentRepacker(embedding_model=embedding_generator)
ollama_summarizer = OllamaSummarizer(api_url="http://localhost:11434/api/chat", model="llama3.2")

def evaluate_rag_pipeline():
    """Evalúa el pipeline RAG usando ground truths y métricas locales."""
    # Cargar ground truths
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Inicializar listas para evaluación
    questions, answers, references, retrieved_contexts = [], [], [], []
    metrics_results = []

    for entry in data["questions"]:
        question = entry["question"]
        ground_truth = entry["ground_truth"]

        print(f"\nEvaluando pregunta: {question}\n")

        # Reformular la pregunta
        rewritten_query = query_rewriter.rewrite(question)
        print(f"Consulta reformulada: {rewritten_query}")

        # Recuperar documentos relevantes con HyDE
        results_with_hyde = hyde.search_with_hyde(rewritten_query, retriever, top_k=10)

        # Reordenar y empaquetar resultados
        repacked_results = repacker.repack(rewritten_query, results_with_hyde)
        reranked_results = reranker.rerank(rewritten_query, repacked_results)

        # Contextos recuperados
        formatted_contexts = [res[0]["text"] if isinstance(res[0], dict) else res[0] for res in reranked_results]
        retrieved_context = " ".join(formatted_contexts)
        retrieved_contexts.append(retrieved_context)

        # Generar respuesta usando modelo local
        response = generate_local_response(prompt=question, max_length=150, temperature=0.7)
        print(f"Respuesta generada: {response}")

        # Calcular métricas
        metrics = {
            "faithfulness": faithfulness(response, retrieved_context),
            "answer_relevancy": answer_relevancy(response, question),
            "context_recall": context_recall(response, retrieved_context),
            "context_precision": context_precision(response, retrieved_context),
        }
        metrics_results.append(metrics)

        # Almacenar datos para evaluación
        questions.append(rewritten_query)
        answers.append(response)
        references.append(ground_truth)

    # Mostrar métricas por pregunta
    for i, metrics in enumerate(metrics_results):
        print(f"\nPregunta {i + 1} métricas: {metrics}")

    # Inicializar acumuladores para las métricas
    total_metrics = {"faithfulness": 0, "answer_relevancy": 0, "context_recall": 0, "context_precision": 0}

    # Calcular el total de cada métrica
    for metrics in metrics_results:
        for key in total_metrics:
            total_metrics[key] += metrics[key]

    # Calcular la media de cada métrica
    num_questions = len(metrics_results)
    average_metrics = {key: total / num_questions for key, total in total_metrics.items()}

    # Mostrar las métricas promedio
    print("Métricas promedio para el conjunto completo de preguntas:")
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.2f}")


if __name__ == "__main__":
    evaluate_rag_pipeline()



