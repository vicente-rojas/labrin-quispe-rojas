import json
import requests

class OllamaSummarizer:
    """
    A class to combine and summarize text fragments using the Ollama API.
    """

    def __init__(self, api_url, model="llama3.2"):
        """
        Initialize the summarizer with the API endpoint and model name.

        :param api_url: URL of the Ollama API endpoint.
        :param model: Model name to use for text generation.
        """
        self.api_url = api_url
        self.model = model

    # def generate_summary(self, reranked_results, num_fragments=5, context="process for obtaining approval for new food additives"):
    #     """
    #     Combine top text fragments and generate a summary using the Ollama API.

    #     :param reranked_results: List of text fragments ranked by relevance.
    #     :param num_fragments: Number of top fragments to combine.
    #     :param context: Context for the summary prompt.
    #     :return: Generated summary as a string.
    #     """
    #     combined_text = " ".join([res[0] for res in reranked_results[:num_fragments]])

    #     prompt = f"""
    #     Using the following extracted text, generate a coherent paragraph summarizing the {context}:

    #     {combined_text}
    #     """

    #     payload = {
    #         "model": self.model,
    #         "messages": [{"role": "user", "content": prompt}]
    #     }

    #     try:
    #         response = requests.post(self.api_url, json=payload, stream=True)
    #         if response.status_code == 200:
    #             generated_paragraph = ""
    #             for line in response.iter_lines(decode_unicode=True):
    #                 try:
    #                     response_json = json.loads(line)
    #                     generated_paragraph += response_json.get("message", {}).get("content", "")
    #                 except json.JSONDecodeError:
    #                     continue
    #             return generated_paragraph.strip()
    #         else:
    #             raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
    #     except requests.RequestException as e:
    #         raise ConnectionError(f"Connection error with Ollama API: {e}")


    def generate_summary(self, reranked_results, num_fragments=5, context=None):
        """
        Combine top text fragments and generate a summary using the Ollama API.

        :param reranked_results: List of text fragments ranked by relevance.
        :param num_fragments: Number of top fragments to combine.
        :param context: Context for the summary prompt. If None, use a default context.
        :return: Generated summary as a string.
        """
        combined_text = " ".join([res[0] for res in reranked_results[:num_fragments]])
        
        # Use a default context if none is provided
        if not context:
            context = "process for obtaining approval for new food additives"
        
        prompt = f"""
        Using the following extracted text, generate a coherent paragraph summarizing the {context}:

        {combined_text}
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True)
            if response.status_code == 200:
                generated_paragraph = ""
                for line in response.iter_lines(decode_unicode=True):
                    try:
                        response_json = json.loads(line)
                        generated_paragraph += response_json.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue
                return generated_paragraph.strip()
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            raise ConnectionError(f"Connection error with Ollama API: {e}")
