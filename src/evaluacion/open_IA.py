# import openai
# class OpenAIChat:
#     def __init__(self, model, api_key, temperature=0, max_tokens=512):
#         self.model = model
#         self.api_key = api_key
#         self.temperature = temperature
#         self.max_tokens = max_tokens

#     def generate(self, prompt):
#         openai.api_key = self.api_key
#         response = openai.ChatCompletion.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=self.temperature,
#             max_tokens=self.max_tokens
#         )
#         return response["choices"][0]["message"]["content"]

import openai

class OpenAIChat:
    def __init__(self, model, api_key, temperature=0, max_tokens=512):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def set_run_config(self, run_config):
        # Configuración adicional si es requerida
        self.run_config = run_config

    def generate(self, prompt):
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response["choices"][0]["message"]["content"]
