# Catching up
In this project, we will reflect on the optimizations that can be applied to an existing Qdrant database to balance indexing speed, search speed, and the retrieval accuracy.

We assume you have successfully implemented the Vector database with Qdrant project and have an existing `arxiv_papers` collection to work with. If this is not the case, navigate to the first project and we will see you in a short time!

## Theory

By default, Qdrant uses the HNSW index for fast Approximate Nearest Neighbor (ANN) searches. This method returns results that are close to the actual nearest neighbors, but it is not exact. It is suitable for large datasets where doing an exact search would be expensive.

Exact k-Nearest Neighbor (k-NN) search involves searching the entire dataset to find the true nearest neighbors. While this approach is slower and is not suitable for production use, it provides precise results. The exact k-NN search serves as a baseline for evaluating the accuracy of ANN search, allowing for comparisons of the trade-offs between speed and precision.

In this task, we will evaluate the accuracy of the approximate search for our existing collection by calculating the average precision@k and compare the speed of the exact search and the approximate search with a [small test dataset](../dataset/queries_embeddings.json). This is only a single component of the search evaluation and the results should be interpreted with caution, but it does serve as a good sanity check and a way to understand whether the expectations match the current results.

Precision@k is a metric that measures the quality of the search results and focuses on the relevance of items retrieved during the ANN search, and is calculated as

$$\text{Precision@k}=\frac{∣\text{ANN results}∩\text{Exact results}|}{k}​$$

A value of 1 indicates that all approximate search results are identical to the exact search results. High precision means fewer irrelevant or false-positive items are included. k here corresponds to the number of retrieved items (we use 10). Precision measures the quality of the retrieved results by assessing the proportion of retrieved items that are actually relevant. It answers the question:

"Out of all the items the search retrieved, how many are true nearest neighbors?"

Note that in the context of this project, the precision@k is the same as the accuracy (this is not true in the general case), and they will be used interchangeably here. 

Here is how you can perform the approximate (the default) search with the Qdrant client:

```
from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
k = 10

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

ann_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k
).points

The exact k-NN search is done with the addition of the exact=True into the search_params of the .query_points() method:

knn_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k,
    search_params=models.SearchParams(exact=True),
).points
```

## Objectives

The main goal of this task is to implement a function that calculates precision@10 and displays the average precision, average time for the approximate (default) search (with limit=10), and the average time for the exact search (with the same limit=10).

To calculate the average time, you can adapt the following snippet (we assume the embeddings are being iterated through, so this snippet will calculate the time for a single query):

```
import time

knn_times = []

start_time_knn = time.time()
knn_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,
    limit=k,
    search_params=models.SearchParams(exact=True),
).points
knn_time = time.time() - start_time_knn
knn_times.append(knn_time)
```

To print your results, you can utilize the following formatting function:

```
def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')
```

Here is the series of steps to perform:

* Run the Qdrant Docker image (the same way as it was done in the stage 1 of the Vector database with Qdrant project);

* Download the [test dataset](../dataset/queries_embeddings.json) that will be used for the calculation. It is a JSON file with 100 queries and their corresponding text-embedding-ada-002 embeddings. The queries are stored as the keys, and the embeddings are stored as the values.

* Load the dataset with the script. This can be done as follows:

```
import json 

with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)
```

* For each embedding in the test_dataset, run the exact and the approximate searches, calculate the precision, and log the time for a single query. The precision of a single query can be calculated as

```
ann_ids = set(item.id for item in ann_result)
knn_ids = set(item.id for item in knn_result)  
precision = len(ann_ids.intersection(knn_ids)) / k
```

* Append the logged times and the precision to their corresponding lists.

* Calculate the averages of the obtained logged time and the precision.

* Display the averages and reflect on the obtained results.

## Useful resources 

### Docs
[Precision and recall at K in ranking and recommendations from Evidently](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)     
[Hierarchical Navigable Small Worlds (HNSW) from Pinecone](https://www.pinecone.io/learn/series/faiss/hnsw/)    
[Understanding the approximate nearest neighbor (ANN) algorithm from Elastic](https://www.elastic.co/blog/understanding-ann) 
[kNN search from Elastic doccumentation](https://www.elastic.co/docs/solutions/search/vector/knn)   
[exact = True in models.SearchParams()](https://qdrant.tech/documentation/guides/optimize/?q=exact%3Dtrue#fine-tuning-search-parameters)