from src.embedding.dense_embeddings import DenseEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Retriever:
    def __init__(self, model_name, collection_name, vector_size, 
                 distance_metric=Distance.COSINE, alpha=0.7, chunk_texts=None, query_rewriter=None):
        """
        Se inicializa la clase Retriever, con soporte para métodos híbridos de búsqueda y reformulación de consultas.


        Args:
            model_name (str)                : Nombre del modelo de embeddings densos.
            collection_name (str)           : Nombre de la colección la VD en Qdrant.
            vector_size (int)               : Dimensión de los vectores de embeddings.
            distance_metric (Distance)      : Métrica de distancia para la búsqueda (por defecto COSINE).
            alpha (float)                   : Peso de la búsqueda densa en la combinación (0 <= alpha <= 1).
            chunk_texts (list of str)       : Lista de textos para búsquedas sparse.
            query_rewriter (QueryRewriter)  : Instancia del QueryRewriter.
        """
        self.embedding_generator = DenseEmbeddings(model_name=model_name) # Inicializar el generador de embeddings
        self.qdrant_store = QdrantVectorStore(                            # Inicializar la base de datos de vectores
            embedding_model=self.embedding_generator,
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance_metric)
        self.alpha = alpha
        self.tfidf_vectorizer = None
        self.sparse_corpus = []
        self.chunk_texts = chunk_texts
        self.query_rewriter = query_rewriter

    def fit_sparse_index(self, corpus):
        """
        Se define la función ajusta un índice TF-IDF para búsqueda sparse.

        Args:
            corpus (list of str)    : Lista de documentos para indexar el corpus. Se ajusta un vectorizador
                                    TF-IDF al corpus generando una matriz sparce que representa la relevanca
                                    de cada término en cada documento.
        """
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sparse_corpus = self.tfidf_vectorizer.fit_transform(corpus)

    def sparse_search(self, query, top_k=5):
        """
        Se define la funcion realiza una búsqueda sparse (TF-IDF).

        Args:
            query (str): Consulta del usuario/Consulta en lenguaje natural.
            top_k (int): Número de resultados más relevantes a devolver.

        Returns:
            list: Resultados sparse con scores y textos.
        """
        query_vec = self.tfidf_vectorizer.transform([query])                    # Se transforma la consulta a un vector TF-IDF
        scores = np.dot(self.sparse_corpus, query_vec.T).toarray().flatten()    # Se calcula el producto punto entre la consulta y los documentos
        top_indices = np.argsort(scores)[-top_k:][::-1]                         # Se obtienen los índices de los documentos más relevantes

        return [(i, scores[i]) for i in top_indices]                            # Se devuelven los índices y los scores

    def hybrid_search(self, query, top_k=5):
        """
        Se define una función que realiza una búsqueda híbrida combinando una busqueda densa y sparse.

        Args:
            query (str): Consulta del usuario/Consulta en lenguaje natural.
            top_k (int): Número de resultados más relevantes a devolver.

        Returns:
            list: Resultados combinados con scores y payload.
        """
        # Se verifica que los textos existan
        if not self.chunk_texts:
            raise ValueError("chunk_texts no está definido. Asegúrate de pasarlo al inicializar el Retriever.")
        
        # Búsqueda densa en qdrant
        dense_results = self.qdrant_store.search(query, top_k=top_k)
        dense_scores = {res['payload']['text']: res['score'] for res in dense_results}

        # Búsqueda sparse
        sparse_results = self.sparse_search(query, top_k=top_k)

        # Se creaa un diccionario de resultados sparse con los textos como claves
        sparse_scores = {self.chunk_texts[i]: score for i, score in sparse_results}

        # Se combinan los resultados densos y sparse
        combined_results = {}
        for text, score in dense_scores.items():
            combined_results[text] = self.alpha * score
        for text, score in sparse_scores.items():
            if text in combined_results:
                combined_results[text] += (1 - self.alpha) * score
            else:
                combined_results[text] = (1 - self.alpha) * score

        # Se re-ordenan los resultados combinados
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search(self, query, top_k=5, method="hybrid"):
        """
        Se define una funcion que realiza una búsqueda con soporte para Query Rewriting.

        Args:
            query (str): Consulta usuario / Consulta en lenguaje natural.
            top_k (int): Número de resultados más relevantes a devolver.
            method (str): Método de búsqueda ('dense', 'sparse', 'hybrid').

        Returns:
            list: Resultados con scores y payloads asociados.
        """
        # Se reformula la consulta si hay definido un query-rewriter
        if self.query_rewriter:
            query = self.query_rewriter.rewrite(query)
            print(f"Consulta reformulada: {query}")

        if method == "dense":
            return self.qdrant_store.search(query, top_k=top_k)
        elif method == "sparse":
            return self.sparse_search(query, top_k=top_k)
        elif method == "hybrid":
            return self.hybrid_search(query, top_k=top_k)
        else:
            raise ValueError("Método de búsqueda no soportado: elige 'dense', 'sparse' o 'hybrid'.")
















