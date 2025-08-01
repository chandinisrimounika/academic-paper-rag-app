import os
import re
import time
import fitz  # PyMuPDF
import faiss
import numpy as np
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline

# === Configuration ===
EMBED_MODEL = 'all-MiniLM-L6-v2'
EMBED_DIM = 384  # for MiniLM model

# === Load Embedding Model ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()

# === FAISS Setup ===
faiss_index = faiss.IndexFlatL2(EMBED_DIM)
doc_store = []

# === PDF Text Extraction ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc) # type: ignore

# === Reference Extraction + Year Extraction ===
def extract_year(line):
    match = re.search(r'(\d{4})', line)
    if match:
        return int(match.group(1))
    return None

def extract_references(text):
    references = []
    years = []
    ref_section = re.split(r'\bReferences\b|\bREFERENCES\b', text)
    if len(ref_section) > 1:
        lines = ref_section[1].strip().split('\n')
        for line in lines:
            if len(line.strip()) > 30 and re.search(r'\d{4}', line):
                references.append(line.strip())
                years.append(extract_year(line.strip()))
    return references, years

# === Chunking Function ===
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# === Index Chunks in FAISS ===
def index_paper(paper_id, chunks):
    embeddings = embedder.encode(chunks)
    faiss_index.add(np.array(embeddings).astype('float32')) # type: ignore
    doc_store.extend(chunks)

# === Query Retrieval from FAISS ===
def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query])
    D, I = faiss_index.search(np.array(query_vec).astype('float32'), top_k) # type: ignore
    return [doc_store[i] for i in I[0]]

# === Load QA Model ===
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(query, context):
    prompt = f"Context: {context}\nQuestion: {query}"
    result = qa_pipeline(prompt, max_length=256)
    return result[0]['generated_text']

# === Citation Graph ===
citation_graph = nx.DiGraph()

def add_citations(paper, cited_papers):
    for cited in cited_papers:
        citation_graph.add_edge(paper, cited)

# === Seminal Work Influence Scoring ===
def compute_influence_scores():
    scores = nx.pagerank(citation_graph)
    in_degrees = dict(citation_graph.in_degree())
    influence = {}
    for node in citation_graph.nodes():
        influence[node] = round(scores.get(node, 0), 4) + 0.1 * in_degrees.get(node, 0)
    return sorted(influence.items(), key=lambda x: x[1], reverse=True)

# === Clustering for Cross-Disciplinary Discovery ===
def cluster_documents(n_clusters=3):
    if len(doc_store) == 0:
        return None
    embeddings = embedder.encode(doc_store)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots()
    for i in range(n_clusters):
        idxs = [j for j, lbl in enumerate(labels) if lbl == i]
        ax.scatter(reduced[idxs, 0], reduced[idxs, 1], label=f"Cluster {i+1}")
    ax.set_title("📚 Document Clusters (Cross-Disciplinary Insight)")
    ax.legend()
    return fig

# === Streamlit UI ===
st.set_page_config(page_title="Academic Paper RAG", layout="wide")
st.title("📚 Academic Paper RAG with Citation Network")

uploaded_file = st.file_uploader("Upload an academic paper (PDF)", type="pdf")
if uploaded_file:
    file_path = os.path.join("temp.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.info("📄 Extracting text and indexing...")
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    paper_id = uploaded_file.name.split('.')[0]
    index_paper(paper_id, chunks)
    references, years = extract_references(text)
    add_citations(paper_id, references)
    st.success(f"✅ {paper_id} indexed and citations added!")
    os.remove(file_path)

    # === Trend Visualization ===
    if years:
        year_counts = {}
        for y in years:
            if y:
                year_counts[y] = year_counts.get(y, 0) + 1
        st.subheader("📈 Citation Trend Over Time")
        fig2, ax2 = plt.subplots()
        ax2.bar(sorted(year_counts.keys()), [year_counts[y] for y in sorted(year_counts.keys())])
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Citations")
        st.pyplot(fig2)

    # === Influence Scores ===
    st.subheader("🏆 Top Seminal Works")
    top_influential = compute_influence_scores()[:5]
    for i, (paper, score) in enumerate(top_influential, 1):
        st.markdown(f"{i}. **{paper}** (Score: {round(score, 4)})")

    # === Document Clusters ===
    st.subheader("📚 Cross-Disciplinary Document Clustering")
    fig3 = cluster_documents()
    if fig3:
        st.pyplot(fig3)

# === User Query and RAG Output ===
query = st.text_input("🔍 Ask a question about your indexed papers")
if query:
    start_time = time.time()
    context_chunks = retrieve_context(query)
    retrieval_time = round(time.time() - start_time, 3)
    answer = generate_answer(query, "\n".join(context_chunks))
    st.markdown(f"🧠 **Answer:** {answer}")
    st.markdown(f"⏱️ *Retrieved in {retrieval_time} seconds*")

    # === Citation Graph Visualization ===
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(citation_graph, k=0.5, seed=42)
    nx.draw_networkx_nodes(citation_graph, pos, node_color='lightgreen', node_size=1000, ax=ax)
    nx.draw_networkx_edges(citation_graph, pos, arrowstyle='->', arrowsize=10, edge_color='gray', ax=ax)
    labels = {
        node: node[:40] + "..." if len(node) > 40 else node
        for node in citation_graph.nodes()
    }
    nx.draw_networkx_labels(citation_graph, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title("Citation Network", fontsize=14)
    plt.axis("off")
    st.pyplot(fig)
