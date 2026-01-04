# Scientific Information Retrieval Engine

This project implements a semantic search engine for scientific articles,
combining textual similarity and citation graph structure.

## Objective
Given a query article, the system must retrieve the most semantically related
articles among a set of candidates, using:
- Sparse representations (TF / TF-IDF)
- Dense representations (SBERT)
- Structural information (citation graph)

## Methods
- **SBERT** (`all-MiniLM-L6-v2`) for dense embeddings
- **Cosine similarity** for ranking
- **Citation graph smoothing** to enrich document representations
- Evaluation using Precision@5, Recall@5, F1@5 and AUC

## Results
SBERT baseline:
- F1@5 ≈ 0.81

SBERT + citation graph (α = 0.75):
- F1@5 ≈ 0.83
- AUC ≈ 0.96
