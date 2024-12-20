from sentence_transformers import CrossEncoder

class DocumentReRanker:
    def __init__(self, model_name="sentence-transformers/all-distilroberta-v1"):
        """
        Se inicializa la clase DocumentReRanker, el cual define la funcionalidad de reordenar
        documentos (chunks) en funcion de su relevancia (metrica) de una consulta especifica.
        Args:
            model_name (str): Nombre del modelo de Cross-Encoder para re-ranking.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results):
        """
        La funcion rerank re-rankea (reordena) los documentos recuperados según la relevancia con la consulta.
        Aunque los documentos ya vienen con un score, este metodo recalcula los scores con el modelo
        utilizando Cross-Encoder para cacular nuevas puntuaciones mas precisas en base a la consulta.
        Args
            query (str): Consulta reformulada.
            results (list): Lista de tuplas (texto_documento, score).

        Returns:
            list: Lista reordenada de documentos [(texto, score)].
        """
        # Se crean pares de query-texto_documento para hacer el re-ranking
        query_document_pairs = [(query, doc[0]) for doc in results]

        # Se obtenen las puntuaciones del modelo de re-ranking
        rerank_scores = self.model.predict(query_document_pairs)

        # Finalmente se combinan documentos con nuevas puntuaciones y reordenar
        reranked_results = [
            (results[i][0], rerank_scores[i]) for i in range(len(results))
        ]
        reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
        return reranked_results
