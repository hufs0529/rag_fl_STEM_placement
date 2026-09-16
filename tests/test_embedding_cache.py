import numpy as np
import pytest

from src.embedding_cache import corpus_fingerprint, load_or_encode, texts_digest


def test_digest_separates_boundaries():
    # 구분자 없이 이어 붙이면 두 코퍼스가 같은 해시를 받는다
    assert texts_digest(["ab", "c"]) != texts_digest(["a", "bc"])


def test_fingerprint_changes_with_model_ids_and_texts():
    base = corpus_fingerprint("m", ["1", "2"], texts_digest(["a", "b"]))
    assert corpus_fingerprint("other", ["1", "2"], texts_digest(["a", "b"])) != base
    assert corpus_fingerprint("m", ["2", "1"], texts_digest(["a", "b"])) != base
    assert corpus_fingerprint("m", ["1", "2"], texts_digest(["a", "c"])) != base
    # 같은 입력은 같은 키
    assert corpus_fingerprint("m", ["1", "2"], texts_digest(["a", "b"])) == base


def test_second_call_reuses_the_cache_instead_of_encoding(tmp_path):
    calls = []

    def encode():
        calls.append(1)
        return np.arange(6, dtype=np.float32).reshape(2, 3)

    kwargs = dict(cache_dir=tmp_path, model_name="m",
                  passage_ids=["1", "2"], texts=["a", "b"], log=lambda _m: None)
    first = load_or_encode(encode=encode, **kwargs)
    second = load_or_encode(encode=encode, **kwargs)
    assert len(calls) == 1                       # 두 번째는 디스크에서 읽었다
    np.testing.assert_array_equal(first, second)
    assert second.dtype == np.float32


def test_a_changed_passage_forces_a_re_encode(tmp_path):
    calls = []

    def encode():
        calls.append(1)
        return np.zeros((2, 3), dtype=np.float32)

    load_or_encode(tmp_path, "m", ["1", "2"], ["a", "b"], encode, log=lambda _m: None)
    load_or_encode(tmp_path, "m", ["1", "2"], ["a", "CHANGED"], encode, log=lambda _m: None)
    assert len(calls) == 2                       # 낡은 벡터를 조용히 재사용하지 않는다


def test_a_truncated_cache_file_is_discarded(tmp_path):
    calls = []

    def encode():
        calls.append(1)
        return np.ones((3, 2), dtype=np.float32)

    ids, texts = ["1", "2", "3"], ["a", "b", "c"]
    load_or_encode(tmp_path, "m", ids, texts, encode, log=lambda _m: None)
    path = next(tmp_path.glob("corpus_*.npy"))
    with open(path, "wb") as fh:                 # 행 수가 모자란 파일로 덮어쓴다
        np.save(fh, np.ones((1, 2), dtype=np.float32))
    out = load_or_encode(tmp_path, "m", ids, texts, encode, log=lambda _m: None)
    assert len(calls) == 2
    assert out.shape == (3, 2)


def test_mismatched_ids_and_texts_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_or_encode(tmp_path, "m", ["1"], ["a", "b"], lambda: np.zeros((1, 2)))


# --- 고정 경로 + 사이드카 지문 (Week 2 의 corpus_embeddings.npy) -------------

def test_a_cache_from_a_different_corpus_is_not_reused(tmp_path):
    """실측 사고: 축소 실행(2,000 권)의 벡터가 남은 상태로 운영(200,000 권)을 돌리면
    zip(passages, labels) 가 2,000 으로 조용히 잘려 분할이 잘못 만들어졌다.
    """
    from src.embedding_cache import load_or_encode_at
    path = tmp_path / "corpus_embeddings.npy"
    calls = []

    def enc(n):
        def _f():
            calls.append(n)
            return np.zeros((n, 4), dtype=np.float32)
        return _f

    small_ids = [str(i) for i in range(3)]
    small_txt = [f"t{i}" for i in range(3)]
    load_or_encode_at(path, "m", small_ids, small_txt, enc(3), log=lambda _m: None)
    assert calls == [3]

    big_ids = [str(i) for i in range(10)]
    big_txt = [f"t{i}" for i in range(10)]
    out = load_or_encode_at(path, "m", big_ids, big_txt, enc(10), log=lambda _m: None)
    assert calls == [3, 10]                      # 재사용하지 않고 다시 인코딩했다
    assert out.shape == (10, 4)


def test_an_identical_corpus_is_reused(tmp_path):
    from src.embedding_cache import load_or_encode_at
    path = tmp_path / "corpus_embeddings.npy"
    calls = []

    def enc():
        calls.append(1)
        return np.ones((4, 4), dtype=np.float32)

    ids, txt = [str(i) for i in range(4)], [f"t{i}" for i in range(4)]
    kw = dict(model_name="m", passage_ids=ids, texts=txt, log=lambda _m: None)
    load_or_encode_at(path, encode=enc, **kw)
    load_or_encode_at(path, encode=enc, **kw)
    assert calls == [1]                          # 두 번째는 디스크에서


def test_a_changed_model_forces_a_re_encode(tmp_path):
    from src.embedding_cache import load_or_encode_at
    path = tmp_path / "corpus_embeddings.npy"
    calls = []

    def enc():
        calls.append(1)
        return np.ones((4, 4), dtype=np.float32)

    ids, txt = [str(i) for i in range(4)], [f"t{i}" for i in range(4)]
    load_or_encode_at(path, "bge-small", ids, txt, enc, log=lambda _m: None)
    load_or_encode_at(path, "other-model", ids, txt, enc, log=lambda _m: None)
    assert calls == [1, 1]


def test_a_cache_without_its_fingerprint_is_not_trusted(tmp_path):
    """손으로 복사해 온 .npy 는 지문이 없다 - 맞는지 알 수 없으므로 다시 만든다."""
    from src.embedding_cache import load_or_encode_at
    path = tmp_path / "corpus_embeddings.npy"
    with open(path, "wb") as fh:
        np.save(fh, np.ones((4, 4), dtype=np.float32))
    calls = []

    def enc():
        calls.append(1)
        return np.zeros((4, 4), dtype=np.float32)

    load_or_encode_at(path, "m", [str(i) for i in range(4)],
                      [f"t{i}" for i in range(4)], enc, log=lambda _m: None)
    assert calls == [1]


def test_verify_embeddings_match_names_the_mismatch(tmp_path):
    from src.embedding_cache import verify_embeddings_match
    path = tmp_path / "corpus_embeddings.npy"
    with open(path, "wb") as fh:
        np.save(fh, np.ones((2000, 4), dtype=np.float32))
    with pytest.raises(ValueError) as e:
        verify_embeddings_match(path, [str(i) for i in range(200000)])
    msg = str(e.value)
    assert "2,000 vectors" in msg and "200,000 passages" in msg
    assert "reduced-scale run" in msg
    # 행 수가 맞으면 그대로 돌려준다
    assert verify_embeddings_match(path, [str(i) for i in range(2000)]).shape == (2000, 4)
