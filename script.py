import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import StorageContext, load_index_from_storage

# ── API key ───────────────────────────────────────────────────────────────────

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set!")

# ── Models ────────────────────────────────────────────────────────────────────

# Free local embedding model — no API key needed
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

Settings.system_prompt = """
You are a disaster risk analysis assistant.

Your tasks:

- Analyse questionnaires and interviews
- Identify risk patterns
- Provide structured outputs
- Use formal language

Base answers only on provided documents.
"""

# ── Index ─────────────────────────────────────────────────────────────────────

if os.path.exists("storage"):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")

# Groq supports streaming — use streaming for Q&A, blocking for stats
streaming_query_engine = index.as_query_engine(streaming=True)
query_engine = index.as_query_engine(streaming=False)


# ── Analysis section ──────────────────────────────────────────────────────────

mode = st.selectbox(
    "Select analysis type:",
    ["General Question", "Risk Summary", "Weakness Analysis", "Recommendations"]
)

question = st.text_input("Ask a question:")

if question:
    if mode == "Risk Summary":
        prompt = f"Summarise major disaster risks.\n\nQuestion:\n{question}"
    elif mode == "Weakness Analysis":
        prompt = f"Identify weaknesses in disaster management.\n\nQuestion:\n{question}"
    elif mode == "Recommendations":
        prompt = f"Provide recommendations.\n\nQuestion:\n{question}"
    else:
        prompt = question

    st.subheader("Response")
    output_placeholder = st.empty()
    full_text = ""

    try:
        streaming_response = streaming_query_engine.query(prompt)
        for token in streaming_response.response_gen:
            if not token:
                continue
            full_text += token
            output_placeholder.markdown(full_text + "▌")
        output_placeholder.markdown(full_text)
    except Exception as e:
        if not full_text:
            response = query_engine.query(prompt)
            full_text = response.response
            output_placeholder.markdown(full_text)
        else:
            output_placeholder.markdown(full_text)


# ── Statistics section ────────────────────────────────────────────────────────

st.header("Statistics")

st.markdown(
    "Ask a counting question about the documents — e.g. "
    "*How many people identified flooding as a risk?* or "
    "*How many respondents mentioned poor infrastructure?*"
)

stats_question = st.text_input(
    "Your statistics question:",
    placeholder="e.g. How many people answered that flooding is a major risk?"
)

chart_title = st.text_input(
    "Chart title (optional):",
    placeholder="e.g. Respondents by Risk Type"
)

if st.button("Generate Statistics"):
    if not stats_question.strip():
        st.warning("Please enter a question first.")
    else:
        st.write("Analysing documents…")

        stats_prompt = f"""
You are analysing questionnaire or interview documents.

Answer this question by counting occurrences across all documents:
"{stats_question}"

Return ONLY the results as a simple list in this exact format (one item per line):
Label: count

Rules:
- "Label" is a short name for each answer/category/group
- "count" must be a whole number
- No extra text, no explanation, no bullet points, no headers
- If you cannot find countable data, return: No data found

Example output:
Flooding: 12
Infrastructure: 8
Storms: 5
"""

        response = query_engine.query(stats_prompt)
        text = response.response.strip()

        if "no data found" in text.lower():
            st.info("No countable data found for that question. Try rephrasing.")
        else:
            st.subheader("Raw Output")
            st.write(text)

            try:
                lines = text.split("\n")
                labels, values = [], []

                for line in lines:
                    if ":" in line:
                        label, number = line.split(":", 1)
                        label = label.strip()
                        num_str = number.strip().replace(",", "")
                        if num_str.isdigit():
                            labels.append(label)
                            values.append(int(num_str))

                if labels:
                    df = pd.DataFrame({"Label": labels, "Count": values})

                    st.subheader("Graph")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(df["Label"], df["Count"], color="steelblue")

                    for bar in bars:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.2,
                            str(int(bar.get_height())),
                            ha="center", va="bottom", fontsize=10
                        )

                    ax.set_xlabel("Category")
                    ax.set_ylabel("Count")
                    ax.set_title(chart_title if chart_title.strip() else stats_question)
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.write("Could not parse any counts from the output. Try rephrasing your question.")

            except Exception as e:
                st.write(f"Could not generate graph: {e}")