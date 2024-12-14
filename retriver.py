import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama


embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("csv")

vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store = vector_store , embed_model = embed_model)

llm = Ollama(model="llama3.2" , request_timeout = 420.0)


query_engine = index.as_chat_engine(llm = llm)

response = query_engine.chat("who solved most number of incidents")

print(response)