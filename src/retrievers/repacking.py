from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentRepacker:
    def __init__(self, embedding_model):
        """
        Se inicializa la clase DocumentRepacker. Esta clse encapsula la lógica para reorganizar o combinar
        documentos relacionados en función de sus similitudes con una consulta.
        Args:
            embedding_model: Modelo para generar embeddings de texto.
        """
        self.embedding_model = embedding_model

    def repack(self, query, results):
        '''
        Se define la función que combina fragmentos de documentos relacionados en función de la similitud con la consulta.
        Args:
            query (str)     : Consulta del usuario/Consulta en lenguaje natural.
            results (list)  : Lista de tuplas (texto, score) que representan los fragmentos de documentos y sus puntuaciones.
        Returns:
            list: Lista de tuplas (texto, score) que representan los fragmentos de documentos combinados y sus puntuaciones.
        '''
        formatted_results = []
        for doc in results:
            try:
                text = doc.get("payload", {}).get("text", "")                            # Se obtiene el texto del fragmento   
                score = doc.get("score", 0.0)                                            # Se obtiene la puntuación del fragmento
                if text:
                    formatted_results.append((text, score))                              # Se añade el fragmento a la lista de resultados formateados
            except KeyError as e:
                print(f"Error al procesar un resultado: {doc}. Clave faltante: {e}")     # Se imprime un mensaje de error
                continue

        if not formatted_results:                                                        # Si no hay resultados formateados
            print("No se pudieron transformar los resultados al formato esperado. Verifica las entradas.")
            return [(query, 0.0)]  # Devuelve el query original como resultado de fallback

        # Se generan los embeddings para los fragmentos y la consulta
        query_embedding = self.embedding_model.embed_query(query)
        document_embeddings = self.embedding_model.embed_documents([doc[0] for doc in formatted_results])

        # Se calculan similitudes entre los documentos y la consulta
        similarities = cosine_similarity([query_embedding], document_embeddings)[0]
        print("Similitudes calculadas:", similarities)  

        # Ajuste dinamico del umbral para agrupar fragmentos
        threshold = 0.7
        grouped_fragments = []

        while not grouped_fragments and threshold > 0.1:                                # Mientras no haya fragmentos agrupados y el umbral sea mayor a 0.1
            grouped_fragments = [doc[0] for i, doc in enumerate(formatted_results) if similarities[i] >= threshold]     # Se agrupan los fragmentos
            if not grouped_fragments:                                                   # Si no hay fragmentos agrupados
                print(f"No se encontraron fragmentos relacionados con umbral {threshold:.2f}. Reduciendo el umbral...")
                threshold -= 0.1

        if not grouped_fragments:
            print("No se encontraron fragmentos relacionados incluso con el umbral más bajo. Retornando el fragmento con mayor similitud.")
            max_index = np.argmax(similarities)
            return [(formatted_results[max_index][0], formatted_results[max_index][1])]

        # Se combinan los fragmentos relacionados
        combined_text = " ".join(grouped_fragments)

        # Crear un solo documento combinado con el promedio de puntuaciones
        combined_score = np.mean([doc[1] for doc in formatted_results if doc[0] in grouped_fragments])

        return [(combined_text, combined_score)]