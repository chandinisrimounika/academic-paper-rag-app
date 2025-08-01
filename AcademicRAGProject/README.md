# Academic Paper RAG with Citation Network

## Overview
This project is a Retrieval-Augmented Generation (RAG) system for academic papers. It not only retrieves relevant documents but also analyzes citation relationships, research trends, and identifies seminal works in a field.

## Features
- **Academic Paper Processing:** Upload and index PDF papers.
- **Citation Network Analysis:** Extract references and visualize citation graphs.
- **Seminal Work Recognition:** Identify influential papers using graph algorithms.
- **Trend Visualization:** Track citation trends over time.
- **Cross-Disciplinary Clustering:** Cluster documents for cross-domain insights.
- **Context-Aware Q&A:** Ask questions and get answers based on indexed papers.

## Tech Stack
- **Python**
- **Streamlit** (UI)
- **ChromaDB** (Vector database)
- **HuggingFace Sentence Transformers** (Embeddings)
- **NetworkX** (Citation graph)
- **Matplotlib** (Visualizations)
- **PyMuPDF** (PDF text extraction)
- **Scikit-learn** (Clustering, PCA)

## How to Run
1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Start the app:**
    ```bash
    python -m streamlit run app.py
    ```
3. **Upload a PDF and explore the features!**

## Project Structure
```
AcademicRAGProject/
├── app.py
├── index.html
├── requirements.txt
└── README.md
```

## Usage
- Upload academic papers in PDF format.
- View citation trends and influential works.
- Explore document clusters.
- Ask questions about your indexed papers.

## Requirements Met
- Academic paper processing and indexing
- Citation network analysis and mapping
- Seminal work recognition algorithms
- Cross-disciplinary relationship discovery (via clustering)
- Research trend identification and tracking

## To Improve
- Add cross-reference validation
- Implement more advanced trend and temporal analysis
- Integrate evaluation metrics (accuracy, latency, etc.)

**Contributors:**  
CHANDINI SRI MOUNIKA.VISSAMSETTI 
B.Tech, Computer Science & Engineering
SRM University - AP
📧 chandinisrimounika_vissamsetti@srmap.edu.in
