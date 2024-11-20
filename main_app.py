# import streamlit as st

# import streamlit as st
# from src.embedding.dense_embeddings import DenseEmbeddings
# from src.vector_store_client.Qdrant_Vector_Store import QdrantVectorStore
# from qdrant_client.http.models import Distance

# # Configuración Inicial de la app
# st.set_page_config(page_title="Búsqueda Semántica", layout="wide")

# # Inicializacion de los DenseEmbeddings
# embedding_model = DenseEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Inicializar Client QdrantVectorStore  
# qdrant_store = QdrantVectorStore(
#     embedding_model=embedding_model,
#     collection_name="semantic_chunks",
#     vector_size=768,
#     distance=Distance.COSINE)

# # Título de la aplicación
# st.title("Búsqueda Semántica con Qdrant")

# # Entrada de la consulta del usuario
# query = st.text_input("Introduce tu consulta:", placeholder="Ejemplo: ¿Cómo se muestra el etiquetado en un envase?")

# # Botón para realizar la búsqueda
# if st.button("Buscar"):
#     if query.strip():
#         # Realizar consulta en Qdrant
#         results = qdrant_store.search(query, top_k=5)

#         # Mostrar resultados
#         st.subheader("Resultados de Búsqueda")
#         if results:
#             for i, res in enumerate(results):
#                 st.markdown(f"### Resultado {i + 1}")
#                 st.write(f"**Score:** {res['score']}")
#                 st.write(f"**Texto:** {res['payload'].get('text', 'Sin texto disponible')}")
#                 st.write(f"**Metadata:** {res['payload'].get('metadata', 'Sin metadatos disponibles')}")
#                 st.markdown("---")
#         else:
#             st.warning("No se encontraron resultados.")
#     else:
#         st.warning("Por favor, introduce una consulta válida.")


import streamlit as st
from src.retrievers.retriever import Retriever 

# Configuración Inicial de la app
st.set_page_config(page_title="Búsqueda Semántica", layout="wide")

# Crear una instancia de la clase Retriever
retriever = Retriever(
    model_name="sentence-transformers/all-mpnet-base-v2",
    collection_name="semantic_chunks",
    vector_size=768,
    distance_metric="COSINE")

# Título de la aplicación
st.title("Búsqueda Semántica con Qdrant")

# Entrada de la consulta del usuario
query = st.text_input("Introduce tu consulta:", placeholder="Ejemplo: ¿Cómo se muestra el etiquetado en un envase?")

# Botón para realizar la búsqueda
if st.button("Buscar"):
    if query.strip():
        # Usar el método `search` de la clase Retriever
        results = retriever.search(query, top_k=5)

        # Mostrar resultados
        st.subheader("Resultados de Búsqueda")
        if results:
            for i, res in enumerate(results):
                st.markdown(f"### Resultado {i + 1}")
                st.write(f"**Score:** {res['score']}")
                st.write(f"**Texto:** {res['payload'].get('text', 'Sin texto disponible')}")
                st.write(f"**Metadata:** {res['payload'].get('metadata', 'Sin metadatos disponibles')}")
                st.markdown("---")
        else:
            st.warning("No se encontraron resultados.")
    else:
        st.warning("Por favor, introduce una consulta válida.")

