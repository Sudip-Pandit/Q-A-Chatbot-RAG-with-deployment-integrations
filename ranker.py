import numpy as np

def mmr(query_vec, doc_vecs, lambda_mult=0.4, k=8):
    selected, candidates = [], list(range(len(doc_vecs)))
    if len(doc_vecs) == 0:
        return []
    sim_to_query = np.dot(doc_vecs, query_vec.T).flatten()
    while len(selected) < min(k, len(doc_vecs)):
        if not selected:
            i = int(np.argmax(sim_to_query))
            selected.append(i)
            candidates.remove(i)
            continue
        selected_vecs = doc_vecs[selected]
        diversity = np.max(np.dot(selected_vecs, doc_vecs.T), axis=0)
        mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * diversity
        # exclude already selected
        mmr_score[selected] = -np.inf
        i = int(np.argmax(mmr_score))
        selected.append(i)
        if i in candidates:
            candidates.remove(i)
    return selected
