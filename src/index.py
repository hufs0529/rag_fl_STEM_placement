"""클라이언트별 Qdrant 컬렉션 (Subtask 1.2).

각 클라이언트는 자기 사설 코퍼스에 대한 **자기 인덱스**만 갖는다. 중앙 인덱스를
두면 실험이 전제하는 배포 상황(각 기관이 자기 코퍼스로 RAG 를 돌린다)과
어긋나고, 클라이언트가 다른 기관의 증거를 검색할 수 있게 되어 주제 분리도
깨진다.

search() 시그니처는 src/retrieval.BruteForceRetriever 와 동일하므로, Week 1
게이트 코드와 캐시 생성 코드가 구현 교체 없이 그대로 돌아간다.

Per-client Qdrant collections: each client indexes only its own private corpus,
matching the deployment the experiment is modelled on. The search() signature
matches BruteForceRetriever so gate and cache code is implementation-agnostic.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def collection_name(client_id: int) -> str:
    return f"client_{client_id:02d}"


class QdrantIndex:
    """로컬 on-disk 모드. 시뮬레이션이므로 서버를 띄우지 않는다."""

    def __init__(self, path: str, dim: int = 384, distance: str = "cosine"):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=path)
        self.dim = dim
        self.distance = distance

    def create(self, name: str, recreate: bool = True) -> None:
        from qdrant_client.models import Distance, VectorParams

        distance = {"cosine": Distance.COSINE, "dot": Distance.DOT,
                    "euclid": Distance.EUCLID}[self.distance]
        if recreate:
            self.client.delete_collection(name) if self._exists(name) else None
        if not self._exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.dim, distance=distance),
            )

    def _exists(self, name: str) -> bool:
        return any(c.name == name for c in self.client.get_collections().collections)

    def upsert(
        self,
        name: str,
        passage_ids: Sequence[str],
        vectors: "np.ndarray",
        batch_size: int = 1024,
    ) -> int:
        from qdrant_client.models import PointStruct

        vectors = np.asarray(vectors, dtype=np.float32)
        for start in range(0, len(passage_ids), batch_size):
            chunk_ids = passage_ids[start : start + batch_size]
            points = [
                PointStruct(
                    id=start + i,                       # 컬렉션 내 순번 (pid 는 payload 에)
                    vector=vectors[start + i].tolist(),
                    payload={"pid": pid},
                )
                for i, pid in enumerate(chunk_ids)
            ]
            self.client.upsert(collection_name=name, points=points)
        return len(passage_ids)

    def close(self) -> None:
        self.client.close()


class QdrantRetriever:
    """BruteForceRetriever 와 같은 search() 인터페이스."""

    def __init__(self, index: QdrantIndex, name: str):
        self.index = index
        self.name = name

    def search(self, query_vectors: "np.ndarray", top_n: int) -> List[List[Tuple[str, float]]]:
        """질의 배치를 한 번에 조회. BruteForceRetriever.search() 와 같은 반환 형식.

        qdrant-client 1.12 에서 `search()`/`search_batch()` 가 폐기되고 1.19 에서
        **제거**됐다. 대체는 `query_points()`/`query_batch_points()` 다. 두 이름을
        모두 더듬는 대신 현재 API 를 쓰고, 없으면 바로 설명과 함께 실패한다 -
        조용히 느린 경로로 빠지면 8 만 질문에서 몇 시간이 날아간다.

        배치 호출을 쓰는 이유도 같다. 질의를 하나씩 보내면 8 만 번 왕복한다.
        """
        vectors = np.asarray(query_vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        client = self.index.client
        if not hasattr(client, "query_batch_points"):
            raise AttributeError(
                "qdrant-client exposes neither query_batch_points nor a supported "
                "search API. requirements.txt pins qdrant-client==1.19.1; a much older "
                "or newer client needs this adapter updated."
            )
        from qdrant_client.models import QueryRequest

        out: List[List[Tuple[str, float]]] = []
        # 한 요청에 너무 많이 담으면 응답이 커지므로 적당히 끊는다.
        chunk = 256
        for start in range(0, len(vectors), chunk):
            requests = [
                QueryRequest(query=v.tolist(), limit=top_n, with_payload=True)
                for v in vectors[start : start + chunk]
            ]
            responses = client.query_batch_points(
                collection_name=self.name, requests=requests
            )
            for response in responses:
                out.append([(h.payload["pid"], float(h.score)) for h in response.points])
        return out


def build_client_index(
    index: QdrantIndex,
    client_id: int,
    passage_ids: Sequence[str],
    vectors: "np.ndarray",
    log=print,
) -> str:
    name = collection_name(client_id)
    index.create(name)
    n = index.upsert(name, list(passage_ids), vectors)
    log(f"  client {client_id}: indexed {n:,} passages into {name}")
    return name
