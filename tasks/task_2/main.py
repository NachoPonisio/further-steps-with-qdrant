import json
import time
from cgi import print_form
from functools import wraps
from typing import List, Optional, Callable, Set

from numpy.ma.extras import average
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint, ExtendedPointId

QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
COLLECTION_NAME: str = 'arxiv_papers'
k = 10

class Result(BaseModel):
    hnsw_ef: int
    avg_precision: float
    avg_query_time_ms: float

client: QdrantClient = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

def time_execution_to_list(target_list: List[float]) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time: float = time.perf_counter()
            result = func(*args, **kwargs)
            end_time: float = time.perf_counter()
            execution_time: float = end_time - start_time
            target_list.append(execution_time)

            return result
        return wrapper
    return decorator

knn_times: List[float] = []
ann_times: List[float] = []

@time_execution_to_list(knn_times)
def get_knn(emb: List[float], top_k: int = k, collection_name: str = COLLECTION_NAME) -> Optional[List[ScoredPoint]]:
    knn: List[ScoredPoint] = client.query_points(
        query=emb,
        collection_name=collection_name,
        limit=top_k,
        search_params=models.SearchParams(exact=True)
    ).points

    return knn if knn else None

@time_execution_to_list(ann_times)
def get_ann(emb: List[float], ef: int,  top_k: int = k, collection_name:str = COLLECTION_NAME) -> Optional[List[ScoredPoint]]:
    ann: List[ScoredPoint] = client.query_points(
        query=emb,
        collection_name=collection_name,
        limit=top_k,
        search_params=models.SearchParams(hnsw_ef=ef)
    ).points

    return ann if ann else None

def evaluate_hnsw_ef(k: int, hnsw_ef_values: list[int], test_dataset: dict[str, list[float]]) -> list[Result]:
    global knn_times
    global ann_times

    ground_truth: dict[str, List[ScoredPoint]] = {}
    results: List[Result] = []
    for key, emb in test_dataset.items():
        knn_result: Optional[List[ScoredPoint]] = get_knn(emb)
        if knn_result:
            ground_truth[key] = knn_result

    for ef in hnsw_ef_values:
        ann_times.clear()
        ann_results: dict[str, List[ScoredPoint]] = {}
        precision: list[float] = []
        for key, emb in test_dataset.items():
            ann_result: Optional[List[ScoredPoint]] = get_ann(emb=emb, ef=ef)
            if ann_result:
                ann_results[key] = ann_result
            else:
                continue
            knn_ids = set(item.id for item in ground_truth.get(key))
            ann_ids = set(item.id for item in ann_result)
            curr_precision: float = len(ann_ids.intersection(knn_ids))/k
            precision.append(curr_precision)
        results.append(Result(hnsw_ef=ef, avg_precision=average(precision), avg_query_time_ms=average(ann_times)*1000))

    return results

def result_formatting(k, ef, avg_precision, avg_ann_time):
    print(f'Current ef: {ef}')
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time:.2f} ms')
    print('-' * 40)


def run():
    hnsw_ef_values: list[int] = [10, 20, 50, 100, 200]
    with open(file="dataset/queries_embeddings.json", mode="r", encoding="utf-8") as file:
        test_dataset: dict[str,list[float]] = json.load(file)
        results = evaluate_hnsw_ef(k=k, hnsw_ef_values=hnsw_ef_values, test_dataset=test_dataset)
        for r in results:
            result_formatting(k, r.hnsw_ef, r.avg_precision, r.avg_query_time_ms)
        file.close()



if __name__ == "__main__":
    run()