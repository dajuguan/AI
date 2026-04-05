"""
Given a set a sentences, quickly find the most similar sentence to a given query sentence.
1. a way to maximize similarity given two similar sentences
    - interface: similarity(sentence1, sentence2) -> float
2. Locality sensitive hashing: a way to restrict the search space for the most relevant vectors.
   - interfaces of LSH:
    - init(onehot-encoded sentences, k) -> None
        - config k bands for controlling the number of candidates returned
        - compare weighted Minhash and non-weighted Minhash
    - query(src) -> List[Tuple[float, str]]
"""
