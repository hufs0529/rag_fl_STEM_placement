"""실행 로그(JSONL) + 게이트 결정(JSON) 기록.

모든 측정값은 파일로 남기고, 분석(Week 5)은 오직 이 파일들만 읽는다.
실험 실행과 분석을 분리해야 재실행 없이 tau 스윕을 다시 돌릴 수 있다.

Run logs (JSONL) and gate decisions (JSON). Every measurement is written to
disk and the Week-5 analysis reads nothing else, so the tau sweep can be redone
without re-running any training.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List


class JsonlLogger:
    """append-only 라운드 로그. 한 줄 = 한 라운드(또는 한 이벤트)."""

    def __init__(self, path: str, meta: Dict[str, Any] = None, fresh: bool = False):
        """fresh=True 면 기존 파일을 버리고 새로 쓴다.

        기본은 append 다. 게이트 로그는 이어 붙이는 것이 맞지만, run 로그는
        그렇지 않다: 실패한 run 이 재실행되면 한 파일에 두 번의 시도가 쌓여
        라운드가 0..7, 0..59 로 되돌아간다. 곡선을 그 순서대로 읽으면 보간이
        망가진다. 재실행 지점에 도달한 run 은 정의상 미완(완료된 run 은
        건너뛴다)이므로 버려도 되는 기록이다.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w" if fresh else "a")
        if meta is not None:
            self.write({"event": "meta", "timestamp": time.time(), **meta})

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def save_json(obj: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_json_default))


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def banner(title: str, width: int = 72) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
