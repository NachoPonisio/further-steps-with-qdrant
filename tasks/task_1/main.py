import json
import time
from functools import wraps
from typing import List, Optional, Callable, Set

from numpy.ma.extras import average
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint, ExtendedPointId

QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
COLLECTION_NAME: str = 'arxiv_papers'
k = 10

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
        search_params=models.SearchParams(
            exact=True,
            quantization=models.QuantizationSearchParams(ignore=True)
        )
    ).points

    return knn if knn else None

@time_execution_to_list(ann_times)
def get_ann(emb: List[float], top_k: int = k, collection_name:str = COLLECTION_NAME) -> Optional[List[ScoredPoint]]:
    ann: List[ScoredPoint] = client.query_points(
        query=emb,
        collection_name=collection_name,
        limit=top_k,
    ).points

    return ann if ann else None

def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


def run():
    precision: list[float] =[]
    with open(file="dataset/queries_embeddings.json", mode="r") as file:
        test_dataset = json.load(file)
        for emb in test_dataset.values():
            ann_results: List[ScoredPoint] = get_ann(emb=emb)
            knn_results: List[ScoredPoint] = get_knn(emb=emb)
            ann_ids: Set[ExtendedPointId] = set(item.id for item in ann_results)
            knn_ids: Set[ExtendedPointId] = set(item.id for item in knn_results)
            current_precision: float =  len(ann_ids.intersection(knn_ids))/k
            precision.append(current_precision)
        file.close()
        avg_precision: float = average(precision)
        avg_ann_time: float = average(ann_times)
        avg_knn_time: float = average(knn_times)
        result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)



if __name__ == "__main__":
    run()