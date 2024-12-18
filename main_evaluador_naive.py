import json
import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.retrievers.retriever_naive import Retriever

# Configuración de paths
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

def evaluate_rag_naive():
    """Evalúa un RAG naive usando las métricas locales."""
    # Cargar datos
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(ALL_CHUNKS_PATH, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    # Inicializar el retriever
    retriever = Retriever(
        model_name="sentence-transformers/all-mpnet-base-v2",
        collection_name="semantic_chunks",
        vector_size=768,
        distance_metric="cosine",
    )

    # Inicializar listas para evaluación
    questions, answers, references, metrics_results = [], [], [], []

    for entry in data["questions"]:
        question = entry["question"]
        ground_truth = entry["ground_truth"]

        print(f"\nEvaluando pregunta: {question}\n")

        # Recuperar documentos relevantes
        results = retriever.search(query=question, top_k=1) # Ojo cambiar de 5 a 1
        retrieved_contexts = " ".join([res["payload"]["text"] for res in results])

        # Generar respuesta
        response = generate_local_response(prompt=question, max_length=150, temperature=0.7)
        print(f"Respuesta generada: {response}")

        # Calcular métricas
        metrics = {
            "faithfulness": faithfulness(response, retrieved_contexts),
            "answer_relevancy": answer_relevancy(response, question),
            "context_recall": context_recall(response, retrieved_contexts),
            "context_precision": context_precision(response, retrieved_contexts),
        }
        metrics_results.append(metrics)

        # Almacenar datos para evaluación
        questions.append(question)
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
    evaluate_rag_naive()