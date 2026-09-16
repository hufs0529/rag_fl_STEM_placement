"""코퍼스 임베딩의 내용 주소 기반(content-addressed) 디스크 캐시.

임베더는 동결이고 코퍼스는 정적이므로, 같은 (모델, passage 집합) 조합의 임베딩은
항상 같은 값이다. 그런데 게이트를 다시 돌릴 때마다 25,000 권을 다시 임베딩하면
CPU 에서 37 분이 그대로 다시 든다 (실측). 설정 하나를 고쳐 재실행하는 일이
잦으므로 캐시가 전체 반복 속도를 지배한다.

캐시 키에는 모델 이름, passage id 순서, **본문 해시**를 모두 넣는다. 한 권이라도
다르면 키가 달라져 자동으로 다시 계산된다. 낡은 벡터를 조용히 재사용하는 것은
다시 임베딩하는 것보다 훨씬 나쁘다 - 틀린 결과가 맞는 결과처럼 보이기 때문이다.

Content-addressed cache for corpus embeddings. The key covers the model name, the
passage id order and a digest of the passage texts, so any change re-encodes
rather than silently reusing stale vectors.
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


def texts_digest(texts: Sequence[str]) -> str:
    """본문 전체의 해시. 구분자를 넣어 ("ab","c") 와 ("a","bc") 가 섞이지 않게 한다."""
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def corpus_fingerprint(model_name: str, passage_ids: Sequence[str], digest: str) -> str:
    h = hashlib.sha256()
    for part in (model_name, str(len(passage_ids)), digest):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    for pid in passage_ids:
        h.update(str(pid).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


def load_or_encode(
    cache_dir,
    model_name: str,
    passage_ids: Sequence[str],
    texts: Sequence[str],
    encode: Callable[[], np.ndarray],
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """캐시에 있으면 읽고, 없으면 encode() 를 돌려 저장한 뒤 돌려준다."""
    if len(passage_ids) != len(texts):
        raise ValueError("passage_ids and texts must be the same length")
    key = corpus_fingerprint(model_name, passage_ids, texts_digest(texts))
    path = Path(cache_dir) / f"corpus_{key}.npy"

    if path.exists():
        vectors = np.load(path)
        if vectors.shape[0] == len(passage_ids):
            log(f"  reusing cached corpus embeddings ({vectors.shape[0]:,} x {vectors.shape[1]}) <- {path}")
            return vectors.astype(np.float32)
        # 키가 맞는데 행 수가 다르면 잘린 파일이다. 버리고 다시 계산한다.
        log(f"  cached file has {vectors.shape[0]:,} rows, expected {len(passage_ids):,}; re-encoding")

    vectors = np.asarray(encode(), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    # 파일 객체로 저장해야 numpy 가 확장자를 덧붙이지 않는다. 다 쓴 뒤 rename 해서
    # 중간에 죽어도 반쪽 파일이 캐시로 남지 않게 한다.
    with open(tmp, "wb") as fh:
        np.save(fh, vectors)
    tmp.replace(path)
    log(f"  cached corpus embeddings -> {path}")
    return vectors
