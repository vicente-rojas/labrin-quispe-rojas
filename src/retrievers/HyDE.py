import requests
import json

class HyDE:
    def __init__(self, ollama_url="http://localhost:11434/api/chat", model_name="llama3.2", embedding_model=None):
        '''
        Se define una clase HyDE, la cual se encarga de generar documentos hipotéticos y realizar busquedas en el Vector Store.
        
        Args:
            ollama_url: URL del servidor de Ollama.
            model_name: Nombre del modelo de lenguaje a utilizar para la generación de documentos hipotéticos.
            embedding_model: Modelo de embeddings para generar embeddings de documentos hipotéticos.
        '''
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.embedding_model = embedding_model

    def generate_hypothetical_document(self, query):
        '''Se define la función generate_hypothetical_document, la cual genera un documento hipotético'''

        # Se construye una carga (payload) para la solicitud POST al servidor de Ollama especificando modelo y mensaje.
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user","content": f"""Please generate a concise and coherent hypothetical document in English, consisting of a maximum of 2 paragraphs (no more than 200 words in total). 
                          The document should provide a clear and informative response to the following query: '{query}'. Ensure that the content is accurate, contextually relevant, and well-structured."""
            }]

        }
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)                    # stream=True para obtener la respuesta línea por línea 
            if response.status_code == 200:                                                         # Verificar si la solicitud fue exitosa (200)
                hypothetical_doc = ""
                for line in response.iter_lines(decode_unicode=True):                               # Iteración sobre cada línea de la respuesta
                    try:
                        response_json = json.loads(line)                                            # Cargar la línea como JSON
                        hypothetical_doc += response_json.get("message", {}).get("content", "")     # Obtener el contenido del mensaje
                    except json.JSONDecodeError:                                                    # Manejo de errores de decodificación JSON
                        continue
                return hypothetical_doc.strip()                                                     # Eliminar espacios en blanco al inicio y al final
            else:
                print(f"Error in Ollama API: {response.status_code} - {response.text}")             # Manejo de errores de solicitud (!=200)
                return ""
        except requests.RequestException as e:                                                      # Manejo de errores de conexión
            print(f"Connection error with Ollama: {e}")
            return ""

    def search_with_hyde(self, query, retriever, top_k=5):
        """
        Se define una funcion de busqueda utilizando documentos hipotéticos generados por HyDE.

        Args:
            query (str)     : Consulta en lenguaje natural.
            retriever       : Objeto para recuperar documentos relevantes.
            top_k (int)     : Número de resultados relevantes a devolver.

        Returns:
            list: Resultados de búsqueda.
        """
        # Generar documento hipotético
        hypothetical_doc = self.generate_hypothetical_document(query)

        # Manejo de errores
        if not hypothetical_doc:
            print("Error al generar el documento hipotético.")
            return []

        # Generar embeddings para el documento hipotético
        hypothetical_embedding = self.embedding_model.embed_query(hypothetical_doc)

        # Realiza la búsqueda (retrieval) utilizando el documento hipotético
        results = retriever.search(hypothetical_doc, top_k=top_k, method="dense")
        return results


