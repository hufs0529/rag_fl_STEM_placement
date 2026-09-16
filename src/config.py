"""YAML 설정 로딩 + 점표기 접근 + overlay 병합.

- load_config()      : base YAML 로딩, dev overlay 를 깊은 병합(deep merge)
- Config.__getattr__ : cfg.retrieval.top_k 처럼 점표기로 접근
- Config.resolve()   : gate 결과처럼 런타임에 확정되는 값(K, S, R)을 덮어쓰기

YAML config loading, dotted access and overlay merging.

- load_config()      : load the base YAML, deep-merge a dev overlay
- Config.__getattr__ : dotted access, e.g. cfg.retrieval.top_k
- Config.resolve()   : overwrite values fixed at runtime by the pilot gates (K, S, R)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config(dict):
    """dict 를 점표기로도 읽을 수 있게 감싼 것 / dict with dotted attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - 오타 진단용
            raise AttributeError(f"config key not found: {name}") from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """cfg.get_path("train.local_steps_per_round") 형태의 안전한 조회."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Dict[str, Any] = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """overlay 의 leaf 값만 base 위에 덮어쓴다(리스트는 통째로 교체).

    Overwrite only overlay leaves onto base; lists are replaced wholesale.
    """
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str = "configs/experiment_config.yaml",
    overlay: Optional[str] = None,
    **overrides: Any,
) -> Config:
    """설정 로딩. overlay 는 dev 축소판, overrides 는 'train.rounds=20' 형태의 CLI 덮어쓰기.

    Load the config. `overlay` is the reduced-scale dev file; `overrides` are
    dotted CLI overrides such as train__rounds=20 (see cli_overrides()).
    """
    cfg = Config(yaml.safe_load(Path(path).read_text()))
    if overlay:
        cfg = Config(deep_merge(cfg, yaml.safe_load(Path(overlay).read_text())))
    for dotted, value in overrides.items():
        cfg.set_path(dotted.replace("__", "."), value)
    return cfg


def cli_overrides(pairs) -> Dict[str, Any]:
    """`--set train.rounds=20 --set treatment.depths=[0,50]` 파싱.

    Parse repeated `--set dotted.key=value` arguments; values go through
    yaml.safe_load so ints, floats, bools and lists all work.
    """
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects dotted.key=value, got: {pair}")
        key, raw = pair.split("=", 1)
        out[key.strip().replace(".", "__")] = yaml.safe_load(raw)
    return out


def add_config_args(parser) -> None:
    """모든 스크립트가 공유하는 설정 관련 인자 / shared config CLI arguments."""
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="configs/dev_config.yaml 을 overlay 로 적용 / apply the reduced-scale overlay",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="설정 개별 덮어쓰기 / override a single config value",
    )


def config_from_args(args) -> Config:
    overlay = "configs/dev_config.yaml" if getattr(args, "dev", False) else None
    return load_config(args.config, overlay=overlay, **cli_overrides(args.overrides))
