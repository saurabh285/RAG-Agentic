import os
import re
import numpy as np
import torch
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai

# === Load environment variables ===
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Streamlit config ===
st.set_page_config(page_title="Agentic RAG (Multi-Domain with Gemini)", layout="wide")

# === Load Gemini LLM ===
@st.cache_resource
def load_llm():
    return genai.GenerativeModel("gemini-1.5-flash")

# === Load vectorstores ===
@st.cache_resource
def load_vectorstores():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs_path = "vectorstores"
    return {
        "finance": FAISS.load_local(os.path.join(vs_path, "finance"), embedding_model, allow_dangerous_deserialization=True),
        "marketing": FAISS.load_local(os.path.join(vs_path, "marketing"), embedding_model, allow_dangerous_deserialization=True),
        "general": FAISS.load_local(os.path.join(vs_path, "general"), embedding_model, allow_dangerous_deserialization=True),
    }, embedding_model

# === Load router model (DistilBERT) ===
@st.cache_resource
def load_router_llm():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

# === Label Encoder ===
@st.cache_resource
def get_label_encoder():
    encoder = LabelEncoder()
    encoder.fit(["finance", "marketing", "general"])
    return encoder

# === Route using DistilBERT (optional) ===
def route_query_llm(query):
    encoder = get_label_encoder()
    labels = encoder.classes_
    inputs = router_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = router_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    predicted_idx = np.argmax(probs)
    return [labels[predicted_idx]]

# === Simple keyword-based routing ===
def route_query_keywords(query: str):
    query = query.lower()
    if any(kw in query for kw in ["revenue", "investment", "profit", "budget", "stock"]):
        return ["finance"]
    elif any(kw in query for kw in ["campaign", "brand", "ads", "promotion", "reach"]):
        return ["marketing"]
    else:
        return ["general"]

# === Complexity detection ===
def is_complex(query: str) -> bool:
    query = query.lower()
    complexity_keywords = [
        "compare", "difference", "similarities", "impact", "timeline", "sequence",
        "multi-step", "across", "evaluate", "suggest", "recommend", "affect",
        "correlation", "pattern", "trend", "pros and cons"
    ]
    multi_question = query.count("?") > 1 or bool(re.search(r"\b(and|or)\b", query))
    has_keywords = any(kw in query for kw in complexity_keywords)
    clause_count = query.count(",") + query.count("and") + query.count("or")
    return sum([has_keywords, multi_question, clause_count > 1]) >= 2

# === Document retrieval ===
def retrieve_docs(query, domains, complex_query=False):
    docs = []
    for domain in domains:
        retriever = vectorstores[domain].as_retriever()
        docs.extend(retriever.get_relevant_documents(query))
        if complex_query:
            docs.extend(retriever.get_relevant_documents("summarize " + query))

    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
        if len(unique_docs) >= 5:
            break
    return unique_docs

# === Answer generation ===
def generate_answer(query, context_docs):
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = f"""
You are a helpful AI assistant. Based on the context below, answer the user's question clearly and concisely.

Context:
{context}

Question: {query}

Answer:
"""
    response = llm.generate_content(prompt)
    return response.text.strip(), context

# === Reflection & refinement ===
def reflect_and_refine(query, context, initial_answer):
    prompt = f"""
You are an expert assistant. Reflect on the following answer. If it’s unclear or incomplete, improve it.

Context: {context}

Question: {query}

Initial Answer: {initial_answer}

Refined Answer:
"""
    response = llm.generate_content(prompt)
    return response.text.strip()

# === Initialize resources ===
llm = load_llm()
vectorstores, embedding_model = load_vectorstores()
router_tokenizer, router_model = load_router_llm()

# === Streamlit UI ===
st.title("Agentic RAG System (Multi-Domain with Gemini 1.5 Flash)")

query = st.text_input("Enter your query:", placeholder="e.g., Compare Q2 revenue and brand campaign performance")
enable_reflection = st.checkbox("Enable Self-Reflection", value=True)

if query:
    domains = route_query_keywords(query)  # or use route_query_llm(query)
    complex_flag = is_complex(query)
    docs = retrieve_docs(query, domains, complex_flag)
    answer, context = generate_answer(query, docs)

    if enable_reflection:
        refined_answer = reflect_and_refine(query, context, answer)
    else:
        refined_answer = answer

    st.subheader("Routing Info")
    st.write(f"**Domain(s):** {', '.join(domains)} | **Complexity:** {'Complex' if complex_flag else 'Simple'}")

    with st.expander("Retrieved Context"):
        st.text(context)

    st.subheader("Initial Answer")
    st.write(answer)

    if enable_reflection:
        st.subheader("Self-Refined Answer")
        st.write(refined_answer)
