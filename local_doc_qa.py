# Original repo for the python library: https://github.com/abetlen/llama-cpp-python
# Install llama-cpp-python with: pip install llama-cpp-python
# Install langchain: pip install langchain
# Install instructor embedding: pip install InstructorEmbedding
# Install torch for instructor embedding: pip install torch

# In this script, we demonstrate how to perform document question answering with langchain and a local LLM with llama-cpp-python.
# For embedding, we use InstructorEmbedding: https://huggingface.co/hkunlp/instructor-base
# For vector store, we use a local transient vector store with Chroma
# For LLM, we use a 4bit quantized wizardLM-7B model: https://gpt4all.io/models/ggml-wizardLM-7B.q4_2.bin

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Pointing LLM to the local model path
llm = LlamaCpp(
    model_path="G:\gpt_4_all_v2\gpt_4_all_v2\\bin\ggml-wizardLM-7B.q4_2.bin"
)

# Load the documents
loader = TextLoader('example.txt', encoding='utf8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings using InstructorEmbedding from HuggingFace
embeddings = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ")

db = Chroma.from_documents(texts, embeddings)

# Since the local LLM has limited token length, retrieve only the top 1 document
retriever = db.as_retriever(search_kwargs={"k": 1})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What are some open source alternatives to GPT?"
result = qa({"query": query})
print(result)