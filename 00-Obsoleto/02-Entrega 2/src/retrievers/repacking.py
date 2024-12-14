# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class DocumentRepacker:
#     def __init__(self, embedding_model):
#         """
#         Inicializa el Document Repacker.

#         Args:
#             embedding_model: Modelo para generar embeddings de texto.
#         """
#         self.embedding_model = embedding_model

#     def repack(self, query, results):
#         """
#         Reorganiza o combina documentos relacionados.

#         Args:
#             query (str): Consulta del usuario.
#             results (list): Lista de resultados [(texto, score)].

#         Returns:
#             list: Lista de documentos reempacados [(texto_combinado, score)].
#         """
#         # Generar embeddings para los fragmentos y la consulta
#         query_embedding = self.embedding_model.encode([query])
#         document_embeddings = self.embedding_model.encode([doc[0] for doc in results])

#         # Calcular similitudes entre los documentos y la consulta
#         similarities = cosine_similarity(query_embedding, document_embeddings)[0]

#         # Agrupar fragmentos relacionados (ejemplo: umbral de similitud >= 0.7)
#         grouped_fragments = []
#         for i, doc in enumerate(results):
#             if similarities[i] >= 0.7:
#                 grouped_fragments.append(doc[0])

#         # Combinar los fragmentos relacionados
#         combined_text = " ".join(grouped_fragments)

#         # Crear un solo documento combinado con el promedio de puntuaciones
#         combined_score = np.mean([doc[1] for doc in results if doc[0] in grouped_fragments])

#         return [(combined_text, combined_score)]

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class DocumentRepacker:
#     def __init__(self, embedding_model):
#         """
#         Inicializa el Document Repacker.

#         Args:
#             embedding_model: Modelo para generar embeddings de texto.
#         """
#         self.embedding_model = embedding_model

#     def repack(self, query, results):
#         """
#         Reorganiza o combina documentos relacionados.

#         Args:
#             query (str): Consulta del usuario.
#             results (list): Lista de resultados [{"score": float, "payload": {"text": str}}].

#         Returns:
#             list: Lista de documentos reempacados [(texto_combinado, score)].
#         """
#         # Transformar resultados al formato esperado [(texto, score)]
#         formatted_results = []
#         for doc in results:
#             try:
#                 text = doc["payload"]["text"]
#                 score = doc["score"]
#                 formatted_results.append((text, score))
#             except KeyError as e:
#                 print(f"Error al procesar un resultado: {doc}. Clave faltante: {e}")
#                 continue

#         if not formatted_results:
#             raise ValueError("No se pudieron transformar los resultados al formato esperado.")

#         # Generar embeddings para los fragmentos y la consulta
#         query_embedding = self.embedding_model.embed_query(query)
#         document_embeddings = self.embedding_model.embed_documents([doc[0] for doc in formatted_results])

#         # Calcular similitudes entre los documentos y la consulta
#         similarities = cosine_similarity([query_embedding], document_embeddings)[0]
#         print("Similitudes calculadas:", similarities)  # Depuración

#         # Agrupar fragmentos relacionados (ejemplo: umbral de similitud >= 0.7)
#         grouped_fragments = []
#         for i, doc in enumerate(formatted_results):
#             if similarities[i] >= 0.3:
#                 grouped_fragments.append(doc[0])

#         if not grouped_fragments:
#             print("No se encontraron fragmentos relacionados. Retornando el fragmento con mayor similitud.")
#             max_index = np.argmax(similarities)
#             return [(formatted_results[max_index][0], formatted_results[max_index][1])]

#         # Combinar los fragmentos relacionados
#         combined_text = " ".join(grouped_fragments)

#         # Crear un solo documento combinado con el promedio de puntuaciones
#         combined_score = np.mean([doc[1] for doc in formatted_results if doc[0] in grouped_fragments])

#         return [(combined_text, combined_score)]

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class DocumentRepacker:
#     def __init__(self, embedding_model):
#         """
#         Se inicializa la clase DocumentRepacker. Esta clse encapsula la lógica para reorganizar o combinar
#         documentos relacionados en función de sus similitudes con una consulta.

#         Args:
#             embedding_model: Modelo para generar embeddings de texto.
#         """
#         self.embedding_model = embedding_model

#     def repack(self, query, results):
#         """
#         Se define la funcion repack, la cual reorganiza o combina documentos relacionados.

#         Args:
#             query (str)     : Consulta del usuario.
#             results (list)  : Lista de resultados [{"score": float, "payload": {"text": str}}].

#         Returns:
#             list: Lista de documentos reempacados [(texto_combinado, score)].
#         """
#         # Se Transforman resultados al formato esperado [(texto, score)]
#         formatted_results = []
#         for doc in results:
#             try:
#                 text = doc["payload"]["text"]
#                 score = doc["score"]
#                 formatted_results.append((text, score))
#             except KeyError as e:
#                 print(f"Error al procesar un resultado: {doc}. Clave faltante: {e}")
#                 continue

#         if not formatted_results:
#             raise ValueError("No se pudieron transformar los resultados al formato esperado.")

#         # Se generan los embeddings para los fragmentos y la consulta
#         query_embedding = self.embedding_model.embed_query(query)
#         document_embeddings = self.embedding_model.embed_documents([doc[0] for doc in formatted_results])

#         # Se calculan similitudes entre los documentos y la consulta
#         similarities = cosine_similarity([query_embedding], document_embeddings)[0]
#         print("Similitudes calculadas:", similarities)  # Depuración

#         # Ajuste dinamico del umbral para agrupar fragmentos
#         threshold = 0.7
#         grouped_fragments = []

#         while not grouped_fragments and threshold > 0.1:
#             grouped_fragments = [doc[0] for i, doc in enumerate(formatted_results) if similarities[i] >= threshold]
#             if not grouped_fragments:
#                 print(f"No se encontraron fragmentos relacionados con umbral {threshold:.2f}. Reduciendo el umbral...")
#                 threshold -= 0.1

#         if not grouped_fragments:
#             print("No se encontraron fragmentos relacionados incluso con el umbral más bajo. Retornando el fragmento con mayor similitud.")
#             max_index = np.argmax(similarities)
#             return [(formatted_results[max_index][0], formatted_results[max_index][1])]

#         # Combinacion de los fragmentos relacionados
#         combined_text = " ".join(grouped_fragments)

#         # Crear un solo documento combinado con el promedio de puntuaciones
#         combined_score = np.mean([doc[1] for doc in formatted_results if doc[0] in grouped_fragments])

#         return [(combined_text, combined_score)]

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
        formatted_results = []
        for doc in results:
            try:
                text = doc.get("payload", {}).get("text", "")
                score = doc.get("score", 0.0)
                if text:
                    formatted_results.append((text, score))
            except KeyError as e:
                print(f"Error al procesar un resultado: {doc}. Clave faltante: {e}")
                continue

        if not formatted_results:
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

        while not grouped_fragments and threshold > 0.1:
            grouped_fragments = [doc[0] for i, doc in enumerate(formatted_results) if similarities[i] >= threshold]
            if not grouped_fragments:
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