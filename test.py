from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


db = chromadb.PersistentClient(path = "./chroma_db")
chroma_collection = db.get_or_create_collection("csv")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents=documents , storage_context=storage_context)