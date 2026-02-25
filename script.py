import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings 
from llama_index.core import StorageContext, load_index_from_storage
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

Settings.system_prompt = """
You are a disaster risk analysis assistant.

Your tasks:

- Analyse questionnaires and interviews
- Identify risk patterns
- Provide structured outputs
- Use formal language

Base answers only on provided documents.
"""

# load docs and create index
if os.path.exists("storage"):

    storage_context = StorageContext.from_defaults(
        persist_dir="storage"
    )

    index = load_index_from_storage(storage_context)

else:

    documents = SimpleDirectoryReader("data").load_data()

    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(
        persist_dir="storage"
    )

# 4. Create query engine
query_engine = index.as_query_engine()


# 5. Select analysis mode
mode = st.selectbox(

    "Select analysis type:",

    [
        "General Question",
        "Risk Summary",
        "Weakness Analysis",
        "Recommendations"
    ]

)

question = st.text_input(
    "Ask a question:"
)

if question:

    if mode == "Risk Summary":

        prompt = f"""
        Summarise major disaster risks.

        Question:
        {question}
        """

    elif mode == "Weakness Analysis":

        prompt = f"""
        Identify weaknesses in disaster management.

        Question:
        {question}
        """

    elif mode == "Recommendations":

        prompt = f"""
        Provide recommendations.

        Question:
        {question}
        """

    else:

        prompt = question

    response = query_engine.query(prompt)

    st.write(response.response)
st.header("Statistics")

if st.button("Generate Risk Statistics"):

    st.write("Analyzing documents...")

    stats_prompt = """
Extract disaster risks and count how often they appear.

Return in this format:

Flooding: number
Infrastructure: number
Storms: number
Health: number
"""

    response = query_engine.query(stats_prompt)

    text = response.response

    st.subheader("Raw Output")

    st.write(text)

    try:

        lines = text.split("\n")

        risks = []
        values = []

        for line in lines:

            if ":" in line:

                name, number = line.split(":")

                risks.append(name.strip())
                values.append(int(number.strip()))

        df = pd.DataFrame({
            "Risk": risks,
            "Count": values
        })

        st.subheader("Graph")

        fig, ax = plt.subplots()

        ax.bar(df["Risk"], df["Count"])

        st.pyplot(fig)

    except:

        st.write("Could not generate graph")