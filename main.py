import pickle
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import *

# charger les données
corpus = load_corpus("corpus.jsonl")
queries = load_queries("queries.jsonl")
qrels = load_qrels("valid.tsv")

# charger les embeddings
with open("corpus_embeddings_sbert.pkl", "rb") as f:
    pack = pickle.load(f)

E = pack["embeddings"]
doc_ids = pack["doc_ids"]
id2idx = {d:i for i,d in enumerate(doc_ids)}
model = SentenceTransformer(pack["model"])

# construire le graph
G = nx.DiGraph()
for d in corpus:
    G.add_node(d)
for d, doc in corpus.items():
    for ref in doc.get("metadata", {}).get("references", []):
        if ref in corpus:
            G.add_edge(d, ref)

# alpha a été choisi par grid search
alpha = 0.75

E_smooth = smooth_sbert_embeddings(E, G, id2idx, alpha=alpha)

print(f"=== SBERT + graphe (alpha={alpha}) ===")
res = evaluate_sbert(qrels, E_smooth, model, id2idx, queries, corpus)
print(res)
