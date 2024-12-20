import requests

class OllamaLLM:
    def __init__(self, api_url, model_name):
        """
        Inicializa el cliente para interactuar con el modelo Ollama.

        Args:
            api_url (str): URL del servidor API de Ollama.
            model_name (str): Nombre del modelo que se utilizará.
        """
        self.api_url = api_url
        self.model_name = model_name

    def set_run_config(self, run_config):
        """
        Configura el entorno de ejecución para este LLM.

        Args:
            run_config (dict): Configuración de ejecución proporcionada por RAGAS.
        """
        # Método necesario para compatibilidad con RAGAS
        self.run_config = run_config

def generate(self, prompt, stop=None, **kwargs):
    payload = {
        "model": self.model_name,
        "prompt": prompt,
    }
    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        generated_text = data.get("text", "")

        return {
            "outputs": [{"output": generated_text}] if generated_text else []
        }
    except requests.RequestException as e:
        raise RuntimeError(f"Error al comunicarse con el servidor Ollama: {e}")


