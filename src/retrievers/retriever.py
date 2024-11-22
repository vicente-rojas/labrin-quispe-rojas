from src.embedding.dense_embeddings import DenseEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore

class Retriever:
    def __init__(self, model_name, collection_name, vector_size, 
                 distance_metric=Distance.COSINE):
        
        """
        Inicializa el Retriever con un modelo de embeddings y una colección de Qdrant.

        Args:
            model_name (str): Nombre del modelo de embeddings.
            collection_name (str): Nombre de la colección en Qdrant.
            vector_size (int): Dimensión de los vectores de embeddings.
            distance_metric (Distance): Métrica de distancia para la búsqueda (por defecto COSINE).

        """

        self.embedding_generator = DenseEmbeddings(model_name=model_name)
        
        self.qdrant_store = QdrantVectorStore(
            embedding_model=self.embedding_generator,
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance_metric)

    def search(self, query, top_k=5):
        """
        Realiza una búsqueda en la base de datos Qdrant usando un query en lenguaje natural.

        Args:
            query (str): Consulta en lenguaje natural.
            top_k (int): Número de resultados más relevantes a devolver.

        Returns:
            list: Lista de resultados, cada uno con su score y payload asociado.
        """
        results = self.qdrant_store.search(query, top_k=top_k)
        return results





















