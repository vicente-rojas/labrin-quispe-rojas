import re
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

class SemanticChunker:
    """
    Esta clase se usa para dividir el documento preprocesado en chunks semanticos basados 
    en la distancia de coseno entre oraciones.
    
    Atributos:
        document (str)                                  : El documento a dividir en chunk semanticos.
        buffer_size (int)                               : Número de oraciones adicionales alrededor de la oracion principal
                                                            para calcular el embbeding combinado.
        model_name (str)                                : Nombre del modelo a usar para generar los embeddings
        sentences (list)                                : Lista de oraciones extraidas del documento
        dense_embedding_model (HuggingFaceEmbeddings)   : El modelo para usar en embeddings
    """

    def __init__(self, document, buffer_size=1, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.document = document
        self.buffer_size = buffer_size
        self.model_name = model_name
        self.sentences = self.split_into_sentences()
        self.dense_embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)


    def split_into_sentences(self):

        """
        Este metodo divide el documento en oraciones usando expresiones regulares

        Retorna:
            list: Lista de diccionarios, cada uno contiene una oración y su índice.
        """

        single_sentences_list = re.split(r"(?<=[.!?])\s+", self.document)

        return [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list) if x]



    def combine_sentences(self):

        """
        Este método combina oraciones en oraciones (chunks) en un rango determinado por el buffer_zise(),
        para crear oraciones extendidas. 

        Flujo:
            1.- Itera sobre cada una de las oraciones (self.sentecnes)
            2.- Combina las oraciones antyeriores desde el indice i - buffer_size hasta i-1
            3.- Agrega la oracion central en el indice actual (i)
            4.- Combina con oraciones posteriores desde el indice i + 1 hasta i + buffer_size
            5.- Almacena la oracion combinada en el diccionario combined_sentences
            6.- Retorna la lista self.sentences, con las oraciones combinadas incluidas.

        Retorna:
            list: Lista de dict{}, cada uno conti
            ene una oración combinada y su índice.
        """

        for i in range(len(self.sentences)):
            combined_sentence = ''

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += self.sentences[j]['sentence'] + ' '
            combined_sentence += self.sentences[i]['sentence']

            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(self.sentences):
                    combined_sentence += ' ' + self.sentences[j]['sentence']

            self.sentences[i]['combined_sentence'] = combined_sentence

        return self.sentences

    def calculate_cosine_distances(self):

        """
        EL metodo de la distancia del coseno calcula las distancias de coseno entre oraciones adyacentes.

        Flujo:
            1.- Combina las oraciones usando combine_sentences().
            2.- Genera embeddings para las oraciones combinadas usando el modelo dense_embedidngs()
            3.- Calcula la similitud coseno entre embeddings de oraciones consecutivas.
            4.- Calcula la similitud de distancia: d = 1 - similaridad
            5.- Almacena la distancia en cada oracion.

        Retorna:
            list: Lista de distancias entre oraciones.
        """

        # Asegura que las oraciones combinadas estén creadas
        self.combine_sentences()

        # Utiliza embed_documents en lugar de encode
        numpy_embeddings = self.dense_embedding_model.embed_documents(
            [x['combined_sentence'] for x in self.sentences])

        for i, sentence in enumerate(self.sentences):
            sentence['combined_sentence_embedding'] = numpy_embeddings[i]

        distances = []
        for i in range(len(self.sentences) - 1):
            embedding_current = self.sentences[i]['combined_sentence_embedding']
            embedding_next = self.sentences[i + 1]['combined_sentence_embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distance = 1 - similarity
            distances.append(distance)
            self.sentences[i]['distance_to_next'] = distance
        return distances



    def split_into_chunks(self, threshold):
        """
        El split_into_chunks() divide el documento en chunks basado en la similaridad coseno.

        Args:
            threshold (int) : Umbral para la distancias del coseno. 
                                    Una distancia mayor al umbral indica un cambio significativo entre dos oraciones.

        Flujo:
            1.- Calcula las distancias coseno entre oraciones.
            2.- Identifica los indices donde la distancia supera el umbral estabolecido.
            3.- Divide las oraciones en chunks entre los puntos de cambio.
            4.- Crea y divide una lista de chunks como cadenas de texto.

        Retorna:
            list: Lista de chunks.


        """
        self.combine_sentences()

        distances = self.calculate_cosine_distances()

        indices_above_thresh = [i for i, distance in enumerate(distances) if distance > threshold]

        start_index = 0
        chunks = []

        for index in indices_above_thresh:
            end_index = index
            group = self.sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(self.sentences):
            combined_text = ' '.join([d['sentence'] for d in self.sentences[start_index:]])
            chunks.append(combined_text)

        return chunks