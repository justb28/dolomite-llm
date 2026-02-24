import streamlit as st
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
# Free embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set!")

llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    token=hf_token,
    use_auth_token=True  # ensures it uses your API key
)

# Tell LlamaIndex to use this model
Settings.llm = llm


# 2. Load your documents
documents = SimpleDirectoryReader("data").load_data()


# 3. Build index
index = VectorStoreIndex.from_documents(documents)


# 4. Create query engine
query_engine = index.as_query_engine()


# 5. Ask questions
question = st.text_input("Ask a question about your data:")
if question:
    response = query_engine.query(question)
    st.subheader("Answer")
    st.write(response.response)
    st.write("---")