from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# class Evaluator:
#     def __init__(self, question, ground_truth, retrieved_summary, retrieved_context, llm):
#         self.dataset = [
#             {
#                 "question": question,
#                 "ground_truth": ground_truth,
#                 "retrieved_summary": retrieved_summary,
#                 "retrieved_context": retrieved_context,
#             }
#         ]
#         self.llm = llm

#     def calculate_metrics(self):
#         result = evaluate(
#             dataset=self.dataset,
#             llm=self.llm,
#             metrics=[
#                 context_precision,
#                 context_recall,
#                 faithfulness,
#                 answer_relevancy,
#             ],
#         )
#         return result.to_pandas().iloc[0].to_dict()

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# class Evaluator:
#     def __init__(self, question, ground_truth, retrieved_summary, retrieved_context, llm, embeddings):
#         """
#         Inicializa la clase Evaluator con los datos necesarios.

#         Args:
#             question (str): Pregunta reformulada.
#             ground_truth (str): Respuesta esperada.
#             retrieved_summary (str): Resumen generado por el sistema.
#             retrieved_context (list): Contextos recuperados.
#             llm: Cliente LLM compatible.
#             embeddings: Cliente de embeddings compatible.
#         """
#         self.dataset = [
#             {
#                 "question": question,
#                 "ground_truth": ground_truth,
#                 "retrieved_summary": retrieved_summary,
#                 "retrieved_context": retrieved_context,
#             }
#         ]
#         self.llm = llm
#         self.embeddings = embeddings

#     def calculate_metrics(self):
#         result = evaluate(
#             dataset=self.dataset,
#             llm=self.llm,
#             embeddings=self.embeddings,  # Pasa las embeddings directamente
#             metrics=[
#                 context_precision,
#                 context_recall,
#                 faithfulness,
#                 answer_relevancy,
#             ],
#         )
#         return result.to_pandas().iloc[0].to_dict()

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# class RagasDataset:
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def get_sample_type(self):
#         # Define el tipo de muestra, por ejemplo, "qa" (pregunta-respuesta)
#         return "qa"

#     def to_dict(self):
#         # Devuelve el dataset como un diccionario
#         return self.dataset


# class Evaluator:
#     def __init__(self, question, ground_truth, retrieved_summary, retrieved_context, llm, embeddings):
#         """
#         Inicializa la clase Evaluator con los datos necesarios.

#         Args:
#             question (str): Pregunta reformulada.
#             ground_truth (str): Respuesta esperada.
#             retrieved_summary (str): Resumen generado por el sistema.
#             retrieved_context (list): Contextos recuperados.
#             llm: Cliente LLM compatible.
#             embeddings: Cliente de embeddings compatible.
#         """
#         self.dataset = RagasDataset([
#             {
#                 "question": question,
#                 "ground_truth": ground_truth,
#                 "retrieved_summary": retrieved_summary,
#                 "retrieved_context": retrieved_context,
#             }
#         ])
#         self.llm = llm
#         self.embeddings = embeddings

#     def calculate_metrics(self):
#         result = evaluate(
#             dataset=self.dataset,
#             llm=self.llm,
#             embeddings=self.embeddings,
#             metrics=[
#                 context_precision,
#                 context_recall,
#                 faithfulness,
#                 answer_relevancy,
#             ],
#         )
#         return result.to_pandas().iloc[0].to_dict()


from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

def perform_evaluation(questions, answers, references, retrieved_contexts, llm, embeddings):
    # Crear dataset compatible con ragas
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "reference": references,
        "retrieved_contexts": retrieved_contexts,
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Evaluación con ragas
    result = evaluate(
        dataset=dataset,
        llm=llm,
        embeddings=embeddings,
        metrics=[
            context_recall,
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )

    return result.to_pandas()





