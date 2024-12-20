import requests
import json

class QueryRewriter:
    def __init__(self, ollama_url="http://localhost:11434/api/chat", model_name="llama3.2"):
        '''
        Se define la clase QueryRewriter, la cual se encarga de reescribir una consulta 
        en inglés de manera concisa y sin contexto adicional.
        Args:
        self.ollama_url     := url del ollama server local.
        self.model_name     := nombre del modelo.
        '''
        self.ollama_url = ollama_url
        self.model_name = model_name
        
    def rewrite(self, query):
        '''
        Se define la funcion rewrite, la cual reescribe una consulta en inglés de manera concisa 
        y sin contexto adicional en formato JSON
        '''
        # Se construye una carga (payload) para la solicitud POST al servidor de Ollama especificando modelo y mensaje.
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": f"Rewrite the following query in English, concisely and without additional context: '{query}'"}]
        }

        # Se realiza la solicitud POST al servidor de Ollama con la carga especificada.
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)                # stream=True para obtener la respuesta línea por línea
            if response.status_code == 200:                                                     # Verificar si la solicitud fue exitosa
                rewritten_query = ""
                for line in response.iter_lines(decode_unicode=True):                           # Iteracion sobre cada línea de la respuesta
                    try:
                        response_json = json.loads(line)                                        # Cargar la línea como JSON
                        rewritten_query += response_json.get("message", {}).get("content", "")  # Obtener el contenido del mensaje
                    except json.JSONDecodeError:                                                # Manejo de errores de decodificación JSON
                        print(f"Invalid JSON line: {line}")
                        continue
                return rewritten_query.strip()
            else:
                print(f"Error in Ollama API: {response.status_code} - {response.text}")         # Manejo de errores de solicitud (!=200)
                return query
        except requests.RequestException as e:                                                  # Manejo de errores de conexión
            print(f"Connection error with Ollama: {e}")
            return query

