"""기저 모델 로딩 + LoRA 부착 + 통신 payload 크기 (Subtask 1.2).

모델 규모는 학습 목표에서 따라 나온다: 모델은 새로운 세계 지식을 습득하는 것이
아니라 "검색된 컨텍스트에 근거한 짧은 추출형 답"을 내도록 배우면 되므로,
LoRA 는 기저 체크포인트에 이미 잠재된 grounding 행동만 조정하면 된다.

LoRA r=8 을 attention projection 에만 적용 -> 학습·통신 파라미터 0.92M,
fp16 기준 클라이언트당 방향당 1.84 MB.

Base model loading, LoRA attachment and communication payload size. Model scale
follows from the stated learning objective: the model must learn to produce a
short extractive answer grounded in retrieved context, not to acquire new world
knowledge, so LoRA need only adapt grounding behaviour already latent in the
base checkpoint.
"""

from typing import Dict, List, Tuple

DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def resolve_device(device: str = None) -> str:
    """device 미지정이면 CUDA 가 있을 때만 쓴다. 지정하면 그대로 따른다."""
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: str, device: str) -> str:
    """CPU 에서는 bfloat16 을 쓰지 않는다.

    bf16 은 GPU 용 형식이고, AMX/AVX512-BF16 이 없는 CPU 에서는 에뮬레이션이라
    float32 보다 오히려 느리다. 정확도 이점도 CPU 에서는 없다.
    """
    if device == "cpu" and dtype in ("bfloat16", "float16"):
        return "float32"
    return dtype


def load_base_model(model_name: str, dtype: str = "bfloat16", device: str = None):
    import torch
    from transformers import AutoModelForCausalLM

    device = resolve_device(device)
    dtype = resolve_dtype(dtype, device)
    torch_dtype = getattr(torch, DTYPES.get(dtype, "float32"))
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    return model.to(device)


def attach_lora(model, lora_cfg):
    """attention projection 에만 LoRA 부착. 이 파라미터만 학습되고 통신된다."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def build_model(cfg, device: str = None):
    """cfg 로부터 (tokenizer, LoRA 부착 모델) 생성 - 모든 진입점이 공유."""
    tokenizer = load_tokenizer(cfg.model.base_model)
    device = resolve_device(device)
    base = load_base_model(cfg.model.base_model, cfg.model.dtype, device)
    model = attach_lora(base, cfg.lora)
    return tokenizer, model


def trainable_parameter_names(model) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def count_trainable(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def payload_bytes(n_trainable: int, dtype: str = "float16") -> int:
    """라운드당 클라이언트당 한 방향 payload 바이트.

    총 통신량 = R x N x payload x 2 (업로드 + 다운로드).
    """
    width = {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]
    return n_trainable * width


def describe_payload(n_trainable: int, num_clients: int, rounds: int, dtype: str = "float16") -> Dict:
    per_direction = payload_bytes(n_trainable, dtype)
    return {
        "trainable_parameters": n_trainable,
        "payload_bytes_per_client_per_direction": per_direction,
        "payload_mb_per_client_per_direction": per_direction / 1e6,
        "total_bytes": rounds * num_clients * per_direction * 2,
        "total_mb": rounds * num_clients * per_direction * 2 / 1e6,
        "formula": "R x N x |AB| x 2",
    }


# 학습 피크 메모리 추정 ------------------------------------------------------
#
# 몇 시간짜리 학습이 OOM 킬러에 죽는 것이 가장 비싼 실패다. 실측으로 프로세스가
# 7.2 GB 까지 올라가 7.6 GB 머신에서 죽었고, 그때 WSL2 배포판 전체가 불안정해져
# 편집기 연결까지 끊겼다. 그래서 시작 전에 추정해 보고 거부한다.

def estimate_training_peak_bytes(
    vocab_size: int,
    batch_size: int,
    seq_length: int,
    n_params: int,
    bytes_per_value: int = 4,
    hidden_size: int = 0,
    n_layers: int = 0,
) -> dict:
    """학습 1 스텝의 피크 메모리를 성분별로 추정한다.

    지배항은 모델이 아니라 **출력층 텐서**다. 은닉 상태는 hidden_size 차원인데
    logits 는 vocab_size 차원이라 (SmolLM2-135M 은 576 -> 49,152, 85 배) 마지막
    한 층의 출력이 30 개 층의 활성값 합보다 크다. 역전파에서는 순전파로 저장한
    logits, 손실 계산의 중간값, logits 의 기울기가 **동시에** 살아 있으므로 3 배로
    잡는다.
    """
    logits = batch_size * seq_length * vocab_size * bytes_per_value
    activations = (
        batch_size * seq_length * hidden_size * bytes_per_value * max(1, n_layers) * 2
        if hidden_size and n_layers else 0
    )
    weights = n_params * bytes_per_value
    return {
        "logits_forward": logits,
        "logits_peak": logits * 3,          # 저장분 + 손실 중간값 + 기울기
        "activations": activations,
        "weights": weights,
        "total": logits * 3 + activations + weights,
    }


def available_memory_bytes() -> int:
    """/proc/meminfo 의 MemAvailable. 읽지 못하면 0 을 돌려 판단을 보류한다."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def memory_guard(estimate: dict, headroom: float = 0.75) -> dict:
    """추정 피크가 가용 메모리의 headroom 배를 넘으면 안전하지 않다고 본다.

    headroom 을 1.0 미만으로 두는 이유: 추정은 파이썬·파이토치 할당자 오버헤드와
    데이터 구조를 포함하지 않고, 가용 메모리도 다른 프로세스와 공유한다.
    """
    available = available_memory_bytes()
    budget = available * headroom
    return {
        "estimated_peak_bytes": estimate["total"],
        "available_bytes": available,
        "budget_bytes": int(budget),
        "safe": bool(available == 0 or estimate["total"] <= budget),
        "unknown": available == 0,
    }


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GB"
