
import json
import re
import unicodedata

class PreprocesamientoJson:

    """
    Se crea la clase de "PreprocesaminetoJson" para limpiar y normalizar la informacion
    extraida de los documentos del paso de loader anterior. 
    """

    def __init__(self, input_file, output_file="cleaned_documents.json"):

        """
        Se inicializa la clase con el archivo de entrada y salida.

        Args:
            input_file (str): Ruta del archivo JSON de entrada.
            output_file (str): Ruta del archivo JSON de salida para los datos limpios.
        """

        self.input_file = input_file
        self.output_file = output_file
        self.documents = []



    def cargar_json(self):

        """
        Se carga el contenido del archivo .json de entrada y lo almacena en 
        el self.document y arroja error si como exception.
        """

        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        except Exception as e:
            print(f"Error al cargar JSON: {e}")



    def limpiar_texto(self, text):

        """
        La funcion realiza la limpieza y normalizacion del texto, conservando 
        la puntuación básica y normalizando espacios.

        Args:
            text (str): Texto a limpiar, el cual es cargado en self.documents

        Returns:
            str: Texto preprocesado, limpio y normlaizado.
        """


        # Normalizacion de acentos y caracteres especiales
        text = unicodedata.normalize("NFKD", text)
        
        # Se mantiene los caracteres alfanumericos, espacios y puntuacion
        text = re.sub(r"[^a-zA-Z0-9\s.,;:!?()/-]", "", text)
        
        # Se trasforma todo a lower_cases
        text = text.lower()
        
        # Se añaden espacio entre numeros y letras y letras y numeros
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)  
        text = re.sub(r"([a-z])(\d+)", r"\1 \2", text)   

        # Se reducen multiples espacioa a un solo espacio
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def procesar_documentos(self):

        """
        Aplica la funcion limpiar_texto a todos los documentos cargados en self.docuemts
        Se aplica la limpieza y normalización a cada documento en el archivo json.
        """

        for doc in self.documents:
            doc["text"] = self.limpiar_texto(doc["text"])



    def guardar_json(self):

        """
        Se guarda los documentos procesados (limpios) en un nuevo archivo json.
        """

        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=4)

            print(f"Documentos limpiados guardados en {self.output_file}")
            
        except Exception as e:
            print(f"Error al guardar JSON: {e}")

    def clean(self):
        """
        Ejecuta todo el proceso de carga, limpieza y guardado.
        """
        self.cargar_json()
        self.procesar_documentos()
        self.guardar_json()
