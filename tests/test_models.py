"""device / dtype 자동 선택 (gate 와 본실험이 공유)."""

from src.models import payload_bytes, resolve_dtype


def test_cpu_never_runs_bfloat16():
    """bf16 은 GPU 용 형식이다. AMX 없는 CPU 에서는 에뮬레이션이라 fp32 보다 느리다."""
    assert resolve_dtype("bfloat16", "cpu") == "float32"
    assert resolve_dtype("float16", "cpu") == "float32"


def test_gpu_keeps_the_configured_dtype():
    assert resolve_dtype("bfloat16", "cuda") == "bfloat16"


def test_float32_passes_through_everywhere():
    assert resolve_dtype("float32", "cpu") == "float32"
    assert resolve_dtype("float32", "cuda") == "float32"


def test_payload_accounting_is_unaffected_by_the_compute_dtype():
    # 통신량은 payload_dtype(fp16) 으로 계산한다 - 계산 정밀도와 무관하다
    assert payload_bytes(920_000, "float16") == 1_840_000


def test_the_adapter_payload_is_a_sum_of_the_two_factors_not_a_product():
    """|A| + |B| 는 합이다. 어댑터는 두 인자 행렬을 **둘 다** 전송하므로
    payload 는 원소 개수의 합이고, 곱으로 읽으면 자리수가 12 개가 된다.

    rank r 로 d x d 를 분해하면 A 는 r x d, B 는 d x r 이다.
    """
    r, d = 8, 576
    a, b = r * d, d * r
    assert payload_bytes(a + b, "float16") == (a + b) * 2 == 18_432
    assert payload_bytes(a + b, "float16") != a * b * 2          # 곱이면 42 MB
