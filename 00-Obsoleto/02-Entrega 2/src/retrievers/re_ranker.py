# from sentence_transformers import CrossEncoder

# class DocumentReRanker:
#     def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
#         """
#         Inicializa el modelo de Document Re-ranking.

#         Args:
#             model_name (str): Nombre del modelo de Cross-Encoder para re-ranking.
#         """
#         self.model = CrossEncoder(model_name)

#     def rerank(self, query, results):
#         """
#         Reordena los documentos recuperados según la relevancia con la consulta.

#         Args:
#             query (str): Consulta reformulada.
#             results (list): Lista de tuplas (texto_documento, score).

#         Returns:
#             list: Lista reordenada de documentos [(texto, score)].
#         """
#         # Crear pares (query, texto_documento) para re-ranking
#         query_document_pairs = [(query, doc[0]) for doc in results]

#         # Obtener puntuaciones del modelo de re-ranking
#         rerank_scores = self.model.predict(query_document_pairs)

#         # Combinar documentos con nuevas puntuaciones y reordenar
#         reranked_results = [
#             (results[i][0], rerank_scores[i]) for i in range(len(results))
#         ]
#         reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
#         return reranked_results

from sentence_transformers import CrossEncoder

class DocumentReRanker:
    def __init__(self, model_name="sentence-transformers/all-distilroberta-v1"):
        """
        Se inicializa la clase DocumentReRanker, el cual define la funcionalidad de reordenar
        documentos en funcion de su relevancia con una consulta.

        Args:
            model_name (str): Nombre del modelo de Cross-Encoder para re-ranking.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results):
        """
        La funcion rerank reordena los documentos recuperados según la relevancia con la consulta.
        Aunque los docuemntos ya vienen con un score, este metodo recalcula los scores con el modelo
        utilizando Cross-Encoder para cacular nuevas puntuaciones mas precisas en base a la consulta.

        Args:
            query (str): Consulta reformulada.
            results (list): Lista de tuplas (texto_documento, score).

        Returns:
            list: Lista reordenada de documentos [(texto, score)].
        """
        # Crear pares (query, texto_documento) para re-ranking
        query_document_pairs = [(query, doc[0]) for doc in results]

        # Obtener puntuaciones del modelo de re-ranking
        rerank_scores = self.model.predict(query_document_pairs)

        # Combinar documentos con nuevas puntuaciones y reordenar
        reranked_results = [
            (results[i][0], rerank_scores[i]) for i in range(len(results))
        ]
        reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
        return reranked_results
