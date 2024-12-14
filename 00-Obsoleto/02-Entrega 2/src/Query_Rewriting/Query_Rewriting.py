# import requests
# import json  # Importar el módulo json

# class QueryRewriter:
#     def __init__(self, ollama_url="http://localhost:11434/api/generate", model_name="llama2"):
#         """
#         Inicializa el Query Rewriter utilizando la API de Ollama.

#         Args:
#             ollama_url (str): URL del endpoint de Ollama.
#             model_name (str): Nombre del modelo a usar en Ollama.
#         """
#         self.ollama_url = ollama_url
#         self.model_name = model_name

#     def rewrite(self, query):
#         """
#         Reformula una consulta utilizando Ollama.

#         Args:
#             query (str): Consulta en lenguaje natural.

#         Returns:
#             str: Consulta reformulada o la original en caso de error.
#         """
#         payload = {
#             "model": self.model_name,
#             "prompt": f"Rewrite the following query to improve its clarity and relevance: '{query}'"
#         }

#         try:
#             response = requests.post(self.ollama_url, json=payload, stream=True)

#             # Verificar si la respuesta es válida
#             if response.status_code == 200:
#                 full_response = ""
#                 for line in response.iter_lines(decode_unicode=True):
#                     if line:  # Ignorar líneas vacías
#                         try:
#                             data = json.loads(line)  # Usar json.loads para cargar la línea
#                             response_text = data.get("response", "").strip()
#                             if response_text and not response_text.lower().startswith(("here are", "to clarify")):
#                                 return response_text  # Devolver la primera línea válida
#                         except json.JSONDecodeError:
#                             print(f"Línea no JSON válida: {line}")
                
#                 return query  # Si no se encuentra una línea válida, devolver la consulta original

#             else:
#                 print(f"Error en Ollama: {response.status_code} - {response.text}")
#                 return query  # Devolver la consulta original en caso de error

#         except requests.RequestException as e:
#             print(f"Error en la conexión con Ollama: {e}")
#             return query  # Devolver la consulta original en caso de error de conexión

#         return query  # Devolver la consulta original si no hay reformulación válida


# import requests
# import json

# class QueryRewriter:
#     def __init__(self, ollama_url="http://localhost:11434/api/generate", model_name="llama3.2"):
#         """
#         Inicializa el Query Rewriter utilizando la API de Ollama.

#         Args:
#             ollama_url (str): URL del endpoint de Ollama.
#             model_name (str): Nombre del modelo a usar en Ollama.
#         """
#         self.ollama_url = ollama_url
#         self.model_name = model_name

#     def rewrite(self, query):
#         """
#         Reformula una consulta utilizando Ollama.

#         Args:
#             query (str): Consulta en lenguaje natural.

#         Returns:
#             str: Consulta reformulada o la original en caso de error.
#         """
#         prompt = (
#             f"Please rewrite the following query as a single, concise, and optimized question "
#             f"for a search engine. Original query: '{query}'"
#         )

#         payload = {
#             "model": self.model_name,
#             "prompt": prompt
#         }

#         try:
#             response = requests.post(self.ollama_url, json=payload, stream=True)

#             if response.status_code == 200:
#                 rewritten_query = ""
#                 for line in response.iter_lines(decode_unicode=True):
#                     if line:  # Ignorar líneas vacías
#                         try:
#                             data = json.loads(line)  # Procesar JSON por línea
#                             response_text = data.get("response", "").strip()
#                             if "I'd be happy" not in response_text:  # Ignorar respuestas genéricas
#                                 rewritten_query += response_text
#                         except json.JSONDecodeError:
#                             print(f"Línea no JSON válida: {line}")

#                 # Procesar y extraer la primera consulta válida
#                 options = [s.strip() for s in rewritten_query.split("\n") if s.strip()]
#                 if options:
#                     return options[0]  # Devuelve la primera consulta significativa

#             else:
#                 print(f"Error en Ollama: {response.status_code} - {response.text}")
#                 return query  # Devuelve la consulta original si hay un error

#         except requests.RequestException as e:
#             print(f"Error en la conexión con Ollama: {e}")
#             return query  # Devuelve la consulta original si hay problemas de conexión

#         return query  # Devuelve la consulta original como respaldo

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# class QueryRewriter:
#     def __init__(self, model_name="t5-small"):
#         """
#         Inicializa un Query Rewriter utilizando un modelo de Hugging Face.

#         Args:
#             model_name (str): Nombre del modelo preentrenado de Hugging Face.
#         """
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # def rewrite(self, query):
#     #     """
#     #     Reformula una consulta utilizando el modelo T5 con un prompt mejorado.

#     #     Args:
#     #         query (str): Consulta en lenguaje natural.

#     #     Returns:
#     #         str: Consulta reformulada.
#     #     """
#     #     input_text = (
#     #         f"Rewrite the following query to make it clear, specific, and optimized for a search engine. "
#     #         f"Ensure it is a complete question and retains all relevant details: \"{query}\""
#     #     )
#     #     inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
#     #     outputs = self.model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
#     #     rewritten_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     #     return rewritten_query

#     def rewrite(self, query):
#         input_text = (
#             f"Rewrite the following query to make it a single, clear, and complete question. "
#             f"Retain all relevant details: \"{query}\""
#         )
#         inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
#         outputs = self.model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
#         rewritten_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Validar que la consulta reformulada sea suficientemente larga
#         if len(rewritten_query.split()) < 5:
#             return query  # Retorna la consulta original si la reformulación no es válida

#         return rewritten_query


# import requests
# import json

# class QueryRewriter:
#     def __init__(self, llama_url="http://localhost:11434/api/generate", model_name="llama3.2"):
#         """
#         Inicializa el Query Rewriter utilizando Llama3.2 a través de un endpoint REST.

#         Args:
#             llama_url (str): URL del endpoint de Llama.
#             model_name (str): Nombre del modelo a usar (Llama3.2).
#         """
#         self.llama_url = llama_url
#         self.model_name = model_name

#     def rewrite(self, query):
#         """
#         Reformula una consulta utilizando Llama3.2 para optimizar su claridad y relevancia.

#         Args:
#             query (str): Consulta en lenguaje natural.

#         Returns:
#             str: Consulta reformulada o la original si ocurre un error.
#         """
#         # Definir el prompt para la reformulación de consulta
#         prompt = (
#         f"Rewrite the following query to make it clearer, more specific, and optimized for retrieval purposes. "
#         f"Ensure the query is concise, retains all important details, and provides additional context if necessary: \"{query}\"")

#         # Crear el payload para el request
#         payload = {
#             "model": self.model_name,
#             "prompt": prompt
#         }

#         try:
#             # Hacer la solicitud al endpoint de Llama3.2
#             response = requests.post(self.llama_url, json=payload, stream=True)

#             # Verificar que la respuesta sea exitosa
#             if response.status_code == 200:
#                 rewritten_query = ""
#                 for line in response.iter_lines(decode_unicode=True):
#                     if line:  # Ignorar líneas vacías
#                         try:
#                             data = json.loads(line)  # Procesar JSON línea por línea
#                             response_text = data.get("response", "").strip()
#                             if response_text:  # Validar que no sea vacío
#                                 rewritten_query += response_text
#                         except json.JSONDecodeError:
#                             print(f"Error al procesar una línea de respuesta: {line}")

#                 # Validar longitud mínima de la consulta reformulada
#                 if len(rewritten_query.split()) >= 5:
#                     return rewritten_query.strip()
#                 else:
#                     print("La consulta reformulada es demasiado corta. Retornando consulta original.")
#                     return query

#             else:
#                 print(f"Error en el endpoint de Llama3.2: {response.status_code} - {response.text}")
#                 return query

#         except requests.RequestException as e:
#             print(f"Error al conectar con el endpoint de Llama3.2: {e}")
#             return query

#         return query


# import requests
# import json

# class QueryRewriter:
#     def __init__(self, ollama_url="http://localhost:11434/api/chat", model_name="llama3.2"):
#         self.ollama_url = ollama_url
#         self.model_name = model_name

#     def rewrite(self, query):
#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": f"Reformule la consulta para que sea clara, específica y optimizada para encontrar fragmentos relevantes. Sin ejemplos adicionales: '{query}'"
# }]
#         }
#         try:
#             response = requests.post(self.ollama_url, json=payload, stream=True)
#             if response.status_code == 200:
#                 # Procesar respuesta línea por línea
#                 rewritten_query = ""
#                 for line in response.iter_lines(decode_unicode=True):
#                     try:
#                         response_json = json.loads(line)
#                         rewritten_query += response_json.get("message", {}).get("content", "")
#                     except json.JSONDecodeError:
#                         print(f"Línea no JSON válida: {line}")
#                         continue
#                 return rewritten_query.strip() if rewritten_query else query
#             else:
#                 print(f"Error en Ollama: {response.status_code} - {response.text}")
#                 return query
#         except requests.RequestException as e:
#             print(f"Error de conexión con Ollama: {e}")
#             return query



import requests
import json

class QueryRewriter:
    def __init__(self, ollama_url="http://localhost:11434/api/chat", model_name="llama3.2"):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def rewrite(self, query):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": f"Rewrite the following query in English, concisely and without additional context: '{query}'"}]
        }
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            if response.status_code == 200:
                rewritten_query = ""
                for line in response.iter_lines(decode_unicode=True):
                    try:
                        response_json = json.loads(line)
                        rewritten_query += response_json.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        print(f"Invalid JSON line: {line}")
                        continue
                return rewritten_query.strip()
            else:
                print(f"Error in Ollama API: {response.status_code} - {response.text}")
                return query
        except requests.RequestException as e:
            print(f"Connection error with Ollama: {e}")
            return query

