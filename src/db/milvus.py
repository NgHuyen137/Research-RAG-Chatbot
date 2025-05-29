import os
from pymilvus import (
  FieldSchema,
  CollectionSchema,
  DataType,
  Collection,
  connections
)


def get_collection(collection_name: str) -> Collection:
  # Connect to Milvus server
  connections.connect(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))

  # Define collection schema
  fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=1024)
  ]
  schema = CollectionSchema(fields)

  # Create collection
  collection = Collection(name=collection_name, schema=schema)

  # Create indexes for vectors
  dense_index = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE"
  }

  sparse_index = {
    "index_type": "SPARSE_INVERTED_INDEX",  
    "metric_type": "IP"
  }

  collection.create_index("sparse", sparse_index)
  collection.create_index("dense", dense_index)
  collection.load()
  
  return collection

collection = get_collection("research_paper_collection")
