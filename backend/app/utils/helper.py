def normalize_score(raw_score: float) -> float:
    # Chroma returns distance → convert to similarity
    if raw_score >= 0 and raw_score <= 2:
        return 1 - raw_score   # reasonable heuristic for cosine distance

    # FAISS cosine similarity already in [-1, 1]
    return raw_score
