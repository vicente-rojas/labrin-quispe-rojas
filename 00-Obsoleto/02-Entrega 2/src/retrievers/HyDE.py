# class HyDE:
#     def __init__(self, generator_model, embedding_model):
#         """
#         Se define la clase HyDE

#         Args:
#             generator_model (dict)  : diccionario que contiene el tokenizer y el modelo de 
#                                         lenguaje generativo.
#             embedding_model         : modelo que genera representaciones vectoriales de texto
#                                         (embeddings) 
#         """
#         self.tokenizer = generator_model["tokenizer"]
#         self.model = generator_model["model"]
#         self.embedding_model = embedding_model

#     def generate_hypothetical_document(self, query):
#         """
#         La funcion genera un documento hipotético que responde a la consulta (query) del usuario.

#         Args:
#             query (str): Consulta del usuario, en formato texto.

#         Returns:
#             str: Documento hipotético generado.
#         """
#         input_text = f"Generate a hypothetical document for the query: '{query}'"
#         inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
#         outputs = self.model.generate(inputs["input_ids"], max_length=200, num_beams=5, early_stopping=True)
#         hypothetical_doc = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return hypothetical_doc

#     def search_with_hyde(self, query, retriever, top_k=5):
#         """
#         La fución realiza una búsqueda basada en embeddings del documento hipotético generado por HyDE.

#         Args:
#             query (str)             : Consulta del usuario.
#             retriever (Retriever)   : Objeto retriever (motor de busqueda) para recuperar objetos relevantes.
#             top_k (int)             : Número de resultados relevantes a devolver.

#         Returns:
#             list: Resultados de búsqueda con HyDE.
#         """
#         try:
#             # Generar documento hipotético con funcion de generación de lenguaje
#             hypothetical_doc = self.generate_hypothetical_document(query)

#             # Generar embeddings para el documento hipotético
#             hypothetical_embedding = self.embedding_model.embed_query(hypothetical_doc)

#             # Realizar búsqueda utilizando el embedding generado
#             results = retriever.search(hypothetical_doc, top_k=top_k, method="dense")

#             return results
#         except Exception as e:
#             print(f"Error durante la búsqueda con HyDE: {e}")
#             return []

import requests
import json

class HyDE:
    def __init__(self, ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=None):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.embedding_model = embedding_model

    # def generate_hypothetical_document(self, query):
    #     payload = {
    #         "model": self.model_name,
    #         "messages": [{"role": "user", "content": f"Genera un documento que responda esta consulta: '{query}'"}]
    #     }
    #     try:
    #         response = requests.post(self.ollama_url, json=payload, stream=True)
    #         if response.status_code == 200:
    #             # Procesar respuesta línea por línea
    #             hypothetical_doc = ""
    #             for line in response.iter_lines(decode_unicode=True):
    #                 try:
    #                     response_json = json.loads(line)
    #                     hypothetical_doc += response_json.get("message", {}).get("content", "")
    #                 except json.JSONDecodeError:
    #                     print(f"Línea no JSON válida: {line}")
    #                     continue
    #             return hypothetical_doc.strip() if hypothetical_doc else ""
    #         else:
    #             print(f"Error en Ollama: {response.status_code} - {response.text}")
    #             return ""
    #     except requests.RequestException as e:
    #         print(f"Error de conexión con Ollama: {e}")
    #         return ""

    def generate_hypothetical_document(self, query):
        # Ajustar el prompt para solicitar el documento en inglés
        payload = {
            "model": self.model_name,
            #"messages": [{"role": "user", "content": f"Generate a brief hypothetical document in English, maximun length 2 parragraphs answering this query: '{query}'"}]
            "messages": [{"role": "user","content": f"""Please generate a concise and coherent hypothetical document in English, consisting of a maximum of 2 paragraphs (no more than 200 words in total). 
                          The document should provide a clear and informative response to the following query: '{query}'. Ensure that the content is accurate, contextually relevant, and well-structured."""
            }]

        }
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            if response.status_code == 200:
                hypothetical_doc = ""
                for line in response.iter_lines(decode_unicode=True):
                    try:
                        response_json = json.loads(line)
                        hypothetical_doc += response_json.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue
                return hypothetical_doc.strip()
            else:
                print(f"Error in Ollama API: {response.status_code} - {response.text}")
                return ""
        except requests.RequestException as e:
            print(f"Connection error with Ollama: {e}")
            return ""

    def search_with_hyde(self, query, retriever, top_k=5):
        """
        Realiza una búsqueda utilizando documentos hipotéticos generados por HyDE.

        Args:
            query (str): Consulta en lenguaje natural.
            retriever: Objeto para recuperar documentos relevantes.
            top_k (int): Número de resultados relevantes a devolver.

        Returns:
            list: Resultados de búsqueda.
        """
        hypothetical_doc = self.generate_hypothetical_document(query)

        if not hypothetical_doc:
            print("Error al generar el documento hipotético.")
            return []

        # Generar embeddings para el documento hipotético
        hypothetical_embedding = self.embedding_model.embed_query(hypothetical_doc)

        # Realizar búsqueda
        results = retriever.search(hypothetical_doc, top_k=top_k, method="dense")
        return results


