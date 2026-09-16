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
