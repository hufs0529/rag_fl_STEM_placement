"""bge-small-en-v1.5 임베딩 래퍼 (Subtask 1.2).

임베더는 **동결(frozen)** 이며 코퍼스도 정적이다. 따라서 임베딩은 한 번만
계산해 캐시하면 되고, 이는 비용 모델의 "일회성 indexing" 항에 대응한다.
질의에는 bge 권장 prefix 를 붙이고 passage 에는 붙이지 않는다.

Frozen embedder over a static corpus, so embeddings are computed once and
cached - the one-off indexing term of the cost model. The bge query prefix is
applied to queries only, never to passages.
"""

from typing import List, Optional, Sequence

import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        device: Optional[str] = None,
        batch_size: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,          # cosine == dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        prefixed = [self.query_prefix + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)


def passage_text_for_embedding(passage: dict) -> str:
    """title 과 본문을 합쳐 임베딩한다 (DPR 관례)."""
    title = (passage.get("title") or "").strip()
    return f"{title}. {passage['text'].strip()}" if title else passage["text"].strip()
