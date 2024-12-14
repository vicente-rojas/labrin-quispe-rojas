from langchain_huggingface import HuggingFaceEmbeddings

class DenseEmbeddings:

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """
        Se define la clase DenseEmbeddings para generar los embeddings de los chunks semanticos
        utilizando HuggingFace.

        Args:
            model_name (str): El nombre del modelo HuggingFace que se utilizará para generar embeddings.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    

    def embed_documents(self, docs):
        """
        Genera embeddings para una lista de documentos o chunks. 
        En este caso, se generan los ambedding a partir de los chunks semanticos de la parte
        N° 3 de chunking del docuemntos json.

        Args:
            docs (list of str): Lista de fragmentos de texto.

        Returns:
            list: Lista de embeddings.
        """
        return [self.embedding_model.embed_query(doc) for doc in docs]

    def embed_query(self, query):
        """
        Genera un embedding para una consulta o texto único.

        Args:
            query (str): Texto de consulta.

        Returns:
            numpy.ndarray: Vector de embedding generado para la consulta.
        """
        
        return self.embedding_model.embed_query(query)
