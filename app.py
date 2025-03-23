import os
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# === CONFIG ===
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["GOOGLE_API_KEY"] = ""  # Replace with your actual key or use secrets manager

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
st.set_page_config(page_title="Agentic RAG (Multi-Domain with Gemini)", layout="wide")

# === Load Gemini Model ===
@st.cache_resource
def load_llm():
    return genai.GenerativeModel("gemini-1.5-flash")

# === Load VectorStores and Embeddings ===
@st.cache_resource
def load_vectorstores():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs_path = "vectorstores"
    return {
        "finance": FAISS.load_local(os.path.join(vs_path, "finance"), embedding_model, allow_dangerous_deserialization=True),
        "marketing": FAISS.load_local(os.path.join(vs_path, "marketing"), embedding_model, allow_dangerous_deserialization=True),
        "general": FAISS.load_local(os.path.join(vs_path, "general"), embedding_model, allow_dangerous_deserialization=True),
    }, embedding_model

llm = load_llm()
vectorstores, embedding_model = load_vectorstores()

# === Keyword-Based Multi-Domain Routing ===
def route_query_keywords(query):
    query = query.lower()
    finance_keywords = ["finance", "revenue", "investment", "profit", "budget", "net income", "expenses"]
    marketing_keywords = ["marketing", "brand", "campaign", "ads", "promotion", "audience", "reach"]

    matched_domains = []
    if any(kw in query for kw in finance_keywords):
        matched_domains.append("finance")
    if any(kw in query for kw in marketing_keywords):
        matched_domains.append("marketing")
    if not matched_domains:
        matched_domains.append("general")
    return matched_domains

# === Complexity Detection ===
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

# === Document Retrieval ===
def retrieve_docs(query, domains, complex_query=False):
    docs = []
    for domain in domains:
        retriever = vectorstores[domain].as_retriever()
        docs.extend(retriever.get_relevant_documents(query))
        if complex_query:
            docs.extend(retriever.get_relevant_documents("summarize " + query))

    # Deduplicate and limit to top 5
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
        if len(unique_docs) >= 5:
            break
    return unique_docs

# === Answer Generation ===
def generate_answer(query, context_docs):
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = f"""
You are a financial analyst. Based on the context below, answer the user's question clearly and concisely.

Context:
{context}

Question: {query}

Answer:
"""
    response = llm.generate_content(prompt)
    return response.text.strip(), context

# === Self-Reflection ===
def reflect_and_refine(query, context, initial_answer):
    reflect_prompt = f"""
You are an expert QA assistant. Given the original question, context, and initial answer, reflect on the quality of the answer. If needed, improve it.

Context: {context}

Question: {query}

Initial Answer: {initial_answer}

Reflection & Improved Answer:
"""
    response = llm.generate_content(reflect_prompt)
    return response.text.strip()

# === Streamlit UI ===
st.title("Agentic RAG System (Multi-Domain with Gemini 1.5 Flash)")

query = st.text_input("Enter your query:", placeholder="e.g., Compare Q2 revenue and brand campaign performance")
enable_reflection = st.checkbox("Enable Self-Reflection", value=True)

if query:
    domains = route_query_keywords(query)
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



# import os
# import re
# from transformers import pipeline
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings

# # === Load the embedding model ===
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # === Load FAISS vector stores ===
# vs_path = "vectorstores"
# print("📦 Loading FAISS vector stores...")

# vectorstores = {
#     "finance": FAISS.load_local(os.path.join(vs_path, "finance"), embedding_model, allow_dangerous_deserialization=True),
#     "marketing": FAISS.load_local(os.path.join(vs_path, "marketing"), embedding_model, allow_dangerous_deserialization=True),
#     "general": FAISS.load_local(os.path.join(vs_path, "general"), embedding_model, allow_dangerous_deserialization=True),
# }

# # === Load HuggingFace LLM (TinyLlama) ===
# print("🧠 Loading TinyLlama model from HuggingFace...")
# llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# # === Router: Detect domain based on keywords ===
# def route_query(query):
#     query = query.lower()
#     if any(word in query for word in ["finance", "revenue", "investment", "profit", "budget"]):
#         return "finance"
#     elif any(word in query for word in ["marketing", "brand", "campaign", "ads", "promotion"]):
#         return "marketing"
#     else:
#         return "general"

# # === Adaptive: Detect if query is complex ===
# def is_complex(query: str) -> bool:
#     query = query.lower()
#     complexity_keywords = [
#         "compare", "difference", "similarities", "impact", "timeline", "sequence",
#         "multi-step", "across", "evaluate", "suggest", "recommend", "affect",
#         "correlation", "pattern", "trend", "pros and cons"
#     ]
#     multi_question = query.count("?") > 1 or bool(re.search(r"\b(and|or)\b", query))
#     has_keywords = any(kw in query for kw in complexity_keywords)
#     clause_count = query.count(",") + query.count("and") + query.count("or")
#     conditions = [has_keywords, multi_question, clause_count > 1]
#     return sum(conditions) >= 2

# # === Retrieve documents ===
# def retrieve_docs(query, domain, complex_query=False):
#     retriever = vectorstores[domain].as_retriever()
#     docs = retriever.get_relevant_documents(query)
#     if complex_query:
#         print("🔁 Adaptive RAG: Retrieving additional context...")
#         docs += retriever.get_relevant_documents("summarize " + query)
#     return docs[:5]  # return top 5 relevant chunks

# # === Generate answer ===
# def generate_answer(query, context_docs):
#     context = "\n\n".join(doc.page_content for doc in context_docs)
#     prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
#     result = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, truncation=True)
#     return result[0]['generated_text']

# # === Run full pipeline ===
# def run_rag_pipeline(query):
#     print(f"\n🔍 User Query: {query}")
    
#     domain = route_query(query)
#     print(f"📍 Routed to domain: {domain}")

#     complex_query = is_complex(query)
#     print(f"⚙️ Query complexity: {'Complex' if complex_query else 'Simple'}")

#     docs = retrieve_docs(query, domain, complex_query)
#     print(f"📚 Retrieved {len(docs)} documents.")

#     answer = generate_answer(query, docs)
#     print("\n💬 Final Answer:")
#     print(answer)

# # === Entry point ===
# if __name__ == "__main__":
#     query = input("📝 Enter your query: ")
#     run_rag_pipeline(query)
