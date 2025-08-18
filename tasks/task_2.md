# Balance the search

In the previous stage, we have observed the differences between k-NN and the approximate search. Let's see how we can impact the search by changing one of the search parameters, `hnsw_ef`, without modifying the index or the underlying embeddings.

## Theory

When a query is performed, the HNSW algorithm starts searching from an entry point in the graph and explores neighboring nodes to find the closest vectors to the query vector. You can read more on the [inner workings of HNSW in the Faiss book chapter at Pinecone](https://www.pinecone.io/learn/series/faiss/hnsw/). The `hnsw_ef` parameter determines how many nodes the algorithm keeps in its priority queue (the list of candidates) during this search process. In the context of HNSW, this parameter is often referred to as ef and stands for "exploration factor". A larger candidate list allows to consider a broader set of potential nearest neighbors.

In this stage, we will explore how different values of the `hnsw_ef` parameter affect the accuracy and the speed of the query execution.

## Objectives

Here, your task is to run the approximate search on the test dataset from the previous stage with the following values of `hnsw_ef`:

```
hnsw_ef_values = [10, 20, 50, 100, 200]
```

Similarly to the previous stage, we will calculate the k-NN (exact) search results and use them as the ground truths to calculate the precision. For the approximate search with a different `hnsw_ef` values, you only have to modify the search_params in the `.query_points()` method of the client:

```
from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

ann_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k,
    search_params=models.SearchParams(hnsw_ef=hnsw_ef) # the modification 
).points
```

Here are the steps to solve this stage:

* Keep the test_dataset from the previous stage.

* Create a function named `evaluate_hnsw_ef` with the following parameters:
    * `k`: The number of nearest neighbors to retrieve for each query, should be set to 10.

    * `hnsw_ef_values`: A list of different hnsw_ef values to test. We assume the following values for this stage:

      * `hnsw_ef_values` = [10, 20, 50, 100, 200]

    * `test_dataset`: The test dataset containing query vectors.

* Perform the exact search with k=10 for every embedding in the test dataset. Extract the IDs of the retrieved points from the exact search and store them in the `ground_truth` dictionary with the query ID as the key (the keys are the natural language questions from the test_dataset)

* For each value in the `hnsw_ef_values`, run the approximate search with k=10 for every embedding in the test_dataset. Like in the previous stage, record the query execution time and append the elapsed times to a list (such that you can see the average time for a query for the different values of `hnsw_ef_values`). Extract the IDs of the retrieved points from the approximate search and retrieve the ground truth IDs for the query from the `ground_truth` dictionary.

* Find the intersection of the approximate search IDs and ground truth IDs, and calculate the precision. Append those precisions to a list (such that you can obtain the average precision for the corresponding value of `hnsw_ef`).

* After processing all queries for the current `hnsw_ef` value, calculate the average precision and the average query time. Make a dictionary with these parameters and append them to a results list. The results list can be in the following format:

```
results = [
    {"hnsw_ef": 10, "avg_precision": 0.5, "avg_query_time_ms": 1.932982444763183},
    {"hnsw_ef": 20, "avg_precision": 0.5, "avg_query_time_ms": 2.932982444763183},
    ...
]
```

* Display the results and reflect on the obtained values. 

## Useful resources 

### Topics
[Hierarchical Navigable Small Worlds](https://hyperskill.org/learn/step/52669)     