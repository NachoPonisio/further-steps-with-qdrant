# Quantization
Before starting this task, adjust and run the following snippet to reset the HNSW config parameters to their default values (since the quantization will interact with the HNSW config parameters and it will affect the speed and the search accuracy):

```
client.update_collection(
     collection_name=COLLECTION_NAME,
     hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=100,
     )
)

while True:
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    if collection_info.status == models.CollectionStatus.GREEN:
        break
```

## Theory

Quantization is a way to reduce the precision of high-dimensional vectors to make the database more efficient in terms of storage and inference speed. It involves converting floating-point numbers to lower-precision types and trying to compress the vectors without loosing too much accuracy.

Qdrant supports 3 types of quantization out of the box (meaning that you don't have to preprocess the embeddings separately, and can quantize them inside of Qdrant itself, and also store the quantized vectors alongside the original ones without much configuration). Here are their brief descriptions:

Scalar quantization reduces the precision of vector elements to a set of predefined values. It's also known as uniform quantization and it maps floating-point numbers to integers like 8-bit. This type is perhaps the most universal one in terms of retaining high accuracy, but it only can speed up the search up to 2 times and reduce the storage by a factor of 4.

Product quantization divides a high-dimensional vector into multiple sub-vectors and quantizes each sub-vector independently. This is the most compressing type out of 3, but it might drop the accuracy significantly. You can read more on the inner working of [product quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) in the Faiss book at Pinecone.

Binary quantization is a type of quantization where each vector component is represented by just 1 bit, either 0 or 1. This is more extreme than other forms of quantization, which might use 4 or 8 bits. Reducing each number to a single bit drastically reduces storage needs and speeds up the queries, but also induces a significant loss in accuracy because so much information is lost. Binary quantization does not work universally well with any [embedding model](https://qdrant.tech/documentation/guides/quantization/#binary-quantization) (although it does seem to work with the text-embedding-ada-002, which we are using, and Cohere embeddings), so binary quantization should be applied carefully and mostly to the tested models that can retain accuracy.

## Objectives

In this task, you will also modify the k-NN vs ANN accuracy and speed function from the previous tasks. Here, we ask you to update the collection with a quantization config and scalar quantization. Here is the code you can use:


```
from qdrant_client import QdrantClient, models
# the initialization code here

client.update_collection(
    collection_name=COLLECTION_NAME,
    optimizer_config=models.OptimizersConfigDiff(),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=False,
        ),
    ),
)
```

Here, we quantize into `int8` type. `quantile=0.99` clips the values beyond the 99th percentile (the top 1% extreme values). This prevents outliers from skewing the quantization range. `always_ram=False` allows the quantized vectors to be stored on disk rather than always loaded into memory (such that quantized data can be loaded from disk as needed instead of residing entirely in memory).

Once you quantize the embeddings, your searches will be in the quantized embeddings by default. In case you wish to disable this behavior, use the following code (this snippet is relevant for the kNN quantization - since we want to retrieve the ground truth values and compare them to the results of the quantized search):

```
result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=k,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(ignore=True) # ignore is False by default
    ),
).points
```

For the ANN search, you can use the following `search_params`:

```
search_params = models.SearchParams(
    quantization=models.QuantizationSearchParams(
        rescore=True,
        oversampling=2.0,
    )
)
```

Run the function two times: first, with `rescore = True`, and then, with `rescore = False`. Observe the results and reflect on their meaning.


## Useful resources 

### Topics
[Quantization from Qdrant](https://qdrant.tech/documentation/guides/quantization/)     