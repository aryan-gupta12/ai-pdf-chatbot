def precision_at_k(retrieved_chunks, relevant_chunk_ids, k):
    retrieved_top_k = retrieved_chunks[:k]

    relevant_retrieved = 0

    for chunk_id in retrieved_top_k:
        if chunk_id in relevant_chunk_ids:
            relevant_retrieved += 1

    return relevant_retrieved / k


def recall_at_k(retrieved_chunks, relevant_chunk_ids, k):
    retrieved_top_k = retrieved_chunks[:k]

    relevant_retrieved = 0

    for chunk_id in retrieved_top_k:
        if chunk_id in relevant_chunk_ids:
            relevant_retrieved += 1

    if len(relevant_chunk_ids) == 0:
        return 0

    return relevant_retrieved / len(relevant_chunk_ids)


def mean_reciprocal_rank(retrieved_chunks, relevant_chunk_ids):
    for index, chunk_id in enumerate(retrieved_chunks):
        if chunk_id in relevant_chunk_ids:
            return 1 / (index + 1)

    return 0