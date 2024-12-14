from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("./data").load_data()

db = chromadb.PersistentClient(path = "./chroma_db")

chroma_collection = db.get_or_create_collection("csv")

vector_store = ChromaVectorStore(chroma_collection= chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


index = VectorStoreIndex.from_documents(documents , storage_context=storage_context)

# index.storage_context.persist(persist_dir="./index")
llm = Ollama(model="llama3.2" , request_timeout = 120.0)

query_engine = index.as_query_engine()
response = query_engine.query("who solved most tickets" , llm = llm)
print(response)


