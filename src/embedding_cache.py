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


# 고정 경로 + 사이드카 지문 ----------------------------------------------------
#
# Week 2 는 임베딩을 data/corpus_embeddings.npy 한 곳에 두고 partition 과 indexing
# 이 공유한다(비용 모델의 "일회성 indexing" 항). 그런데 존재 여부만 검사하면
# **다른 코퍼스의 벡터를 그대로 재사용**한다. 실제로 축소 실행(2,000 권)의 벡터가
# 남아 있는 상태에서 운영(200,000 권)을 돌리면, zip(passages, labels) 가 2,000 으로
# 조용히 잘려 20 만 권 중 2 천 권만으로 분할이 만들어진다 - 오류도 없이.
#
# 그래서 벡터 옆에 지문을 남기고, 재사용 전에 대조한다.

def _meta_path(path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + ".meta.json")


def load_or_encode_at(
    path,
    model_name: str,
    passage_ids: Sequence[str],
    texts: Sequence[str],
    encode: Callable[[], np.ndarray],
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """고정 경로에 임베딩을 두고, 지문이 일치할 때만 재사용한다."""
    import json

    if len(passage_ids) != len(texts):
        raise ValueError("passage_ids and texts must be the same length")
    path = Path(path)
    want = {
        "model": model_name,
        "n": len(passage_ids),
        "digest": corpus_fingerprint(model_name, passage_ids, texts_digest(texts)),
    }
    meta = _meta_path(path)

    if path.exists():
        vectors = np.load(path)
        have = None
        if meta.exists():
            try:
                have = json.loads(meta.read_text())
            except (OSError, ValueError):
                have = None
        if have == want and vectors.shape[0] == len(passage_ids):
            log(f"  reusing cached embeddings {vectors.shape} <- {path}")
            return vectors.astype(np.float32)
        # 왜 못 쓰는지 밝힌다. "다시 임베딩합니다" 만 찍으면 왜 느린지 알 수 없다.
        if have is None:
            why = f"no fingerprint beside it ({meta.name} missing)"
        elif have.get("n") != want["n"]:
            why = f"it holds {have.get('n'):,} rows, this corpus has {want['n']:,}"
        elif have.get("model") != want["model"]:
            why = f"it was built with {have.get('model')}, now {want['model']}"
        else:
            why = "the corpus contents changed"
        log(f"  cached embeddings at {path} cannot be reused: {why}; re-encoding")

    vectors = np.asarray(encode(), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, vectors)
    tmp.replace(path)
    meta.write_text(json.dumps(want, indent=2))
    log(f"  embedded corpus -> {path} {vectors.shape}")
    return vectors


def verify_embeddings_match(path, passage_ids: Sequence[str], log=print) -> "np.ndarray":
    """임베딩을 읽되 행 수가 passage 수와 맞는지 확인한다 (indexing 쪽에서 사용).

    맞지 않으면 IndexError 가 나거나 - 더 나쁘게 - 조용히 잘린 결과가 나온다.
    """
    vectors = np.load(Path(path))
    if vectors.shape[0] != len(passage_ids):
        raise ValueError(
            f"{path} holds {vectors.shape[0]:,} vectors but the corpus has "
            f"{len(passage_ids):,} passages. These are embeddings of a different corpus "
            "(a reduced-scale run leaves them behind). Delete the file and re-run "
            "partition_clients.py, which rebuilds it."
        )
    return vectors
