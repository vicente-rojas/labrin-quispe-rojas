import json
import requests

class OllamaSummarizer:
    """
    Se define una clase para combinar y resumir fragmentos de texto utilizando la API de Ollama.
    Args:
        api_url (str)   : URL del punto final de la API de Ollama.
        model (str)     : Nombre del modelo a utilizar para la generación de texto.  
    """
    def __init__(self, api_url, model="llama3.2"):
        self.api_url = api_url
        self.model = model

    def generate_summary(self, reranked_results, num_fragments=5, context=None):
        """
        Se define la funcion generate_summary para combinar los fragmentos de texto superiores y generar un resumen utilizando la API de Ollama.
        Args:
            reranked_results (list)     : Lista de fragmentos de texto clasificados por relevancia.
            num_fragments (int)         : Número de fragmentos superiores a combinar.
            context (str)               : Contexto para la solicitud de resumen. 
                                            Si es None, se utiliza un contexto predeterminado.        
        Returns:
            str: Resumen generado como una cadena.
        """
        
        # Combinar los fragmentos de texto superiores
        combined_text = " ".join([res[0] for res in reranked_results[:num_fragments]])
        
        # Definir un contexto predeterminado si no se proporciona
        if not context:
            context = "process for obtaining approval for new food additives"
        
        prompt = f"""
        Using the following extracted text, generate a coherent paragraph summarizing the {context}:

        {combined_text}
        """
        # Crear una solicitud a la API de Ollama
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True)                       # Realizar una solicitud POST a la API de Ollama
            if response.status_code == 200:                                                         # Si la solicitud es exitosa
                generated_paragraph = ""                                                            # Inicializar el párrafo generado
                for line in response.iter_lines(decode_unicode=True):                               # Iterar sobre las líneas de la respuesta
                    try:
                        response_json = json.loads(line)                                            # Cargar la línea como JSON
                        generated_paragraph += response_json.get("message", {}).get("content", "")  # Obtener el contenido del mensaje
                    except json.JSONDecodeError:                                                    # Manejar errores de decodificación JSON
                        continue
                return generated_paragraph.strip()                                                  # Devolver el párrafo generado
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")   # Levantar una excepción en caso de error
        except requests.RequestException as e:                                                      # Manejar errores de solicitud
            raise ConnectionError(f"Connection error with Ollama API: {e}")                         # Levantar una excepción de conexión
