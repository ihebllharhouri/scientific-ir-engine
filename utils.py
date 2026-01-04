import json
import numpy as np
import networkx as nx
from typing import Dict
from sklearn.metrics import roc_auc_score


# Chargement des données
def load_corpus(file_path: str):
    corpus = {}
    buffer = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            try:
                doc = json.loads(buffer)
                corpus[str(doc["_id"])] = doc
                buffer = ""
            except json.JSONDecodeError:
                continue
    return corpus

def load_queries(file_path: str) -> Dict[str, Dict]:
    queries = {}
    buffer = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            try:
                doc = json.loads(buffer)
                queries[str(doc["_id"])] = doc
                buffer = ""
            except json.JSONDecodeError:
                continue
    return queries

def load_qrels(file_path: str):
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, cid, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels



# Embeddings
def build_doc_text(doc: dict, max_text_chars: int = 1000) -> str:
    title = doc.get("title", "") or ""
    body = (doc.get("text", "") or "")[:max_text_chars]
    return f"{title} [SEP] {body}" if body else title


def get_query_text(qid, queries, corpus, max_text_chars=1000):
    qdoc = queries.get(qid, corpus.get(qid, {}))
    return build_doc_text(qdoc, max_text_chars)



# SBERT 
def rank_candidates_sbert(qid, qrels, E_rep, model, id2idx, queries, corpus):
    qtext = get_query_text(qid, queries, corpus)
    qemb = model.encode([qtext], normalize_embeddings=True)[0].astype(np.float32)

    cids = [cid for cid in qrels[qid] if cid in id2idx]
    idxs = [id2idx[cid] for cid in cids]

    scores = E_rep[idxs] @ qemb
    order = np.argsort(scores)[::-1]
    return [(cids[i], float(scores[i]), qrels[qid][cids[i]]) for i in order]


def evaluate_sbert(qrels, E_rep, model, id2idx, queries, corpus):
    P5, R5, F5, AUC = [], [], [], []
    for qid in qrels:
        ranked = rank_candidates_sbert(qid, qrels, E_rep, model, id2idx, queries, corpus)
        if not ranked:
            continue
        y_true = [r for _,_,r in ranked]
        y_score = [s for _,s,_ in ranked]

        tp = sum(r for _,_,r in ranked[:5])
        total = sum(y_true)

        p = tp / 5
        r = tp / total if total else 0
        f = 2*p*r/(p+r) if p+r else 0

        P5.append(p); R5.append(r); F5.append(f)
        if len(set(y_true)) == 2:
            AUC.append(roc_auc_score(y_true, y_score))

    return {
        "Precision@5": float(np.mean(P5)),
        "Recall@5": float(np.mean(R5)),
        "F1@5": float(np.mean(F5)),
        "AUC": float(np.mean(AUC)) if AUC else None
    }



# graphe et lissage
def smooth_sbert_embeddings(E, G, id2idx, alpha=0.75):
    E2 = E.copy()
    for did, i in id2idx.items():
        if did not in G:
            continue
        neigh = list(G.successors(did)) + list(G.predecessors(did))
        neigh = [n for n in neigh if n in id2idx]
        if neigh:
            mean = E[[id2idx[n] for n in neigh]].mean(axis=0)
            E2[i] = alpha * E[i] + (1-alpha) * mean
    return E2 / (np.linalg.norm(E2, axis=1, keepdims=True)+1e-12)
