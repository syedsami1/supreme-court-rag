"""Elasticsearch index management for legal case chunks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

from elasticsearch import Elasticsearch, helpers


ES_URL = "http://localhost:9200"
INDEX_NAME = "legal_cases_2024"
INPUT_PATH = Path("data/processed/chunks_with_vectors.jsonl")


INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text", "analyzer": "standard"},
            "chunk_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
            "case_id": {"type": "keyword"},
            "citation": {"type": "keyword"},
            "court": {"type": "keyword"},
            "category": {"type": "keyword"},
            "year": {"type": "keyword"},
            "title": {"type": "text"},
            "judge": {"type": "text"},
            "decision_date": {"type": "date"},
            "chunk_id": {"type": "keyword"},
            "para_index": {"type": "integer"},
            "char_length": {"type": "integer"},
            "embedding_model": {"type": "keyword"},
        }
    }
}


def get_client(es_url: str = ES_URL) -> Elasticsearch:
    client = Elasticsearch(es_url, request_timeout=60)
    if not client.ping():
        raise ConnectionError(f"Could not connect to Elasticsearch at {es_url}")
    return client


def create_index(client: Elasticsearch, index_name: str = INDEX_NAME, recreate: bool = False) -> None:
    if client.indices.exists(index=index_name):
        if recreate:
            client.indices.delete(index=index_name)
        else:
            print(f"Index already exists: {index_name}")
            return

    client.indices.create(index=index_name, body=INDEX_MAPPING)
    print(f"Created index: {index_name}")


def iter_vector_records(input_path: Path = INPUT_PATH) -> Iterable[Dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Vector chunk file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _bulk_actions(index_name: str, records: Iterable[Dict]) -> Iterable[Dict]:
    for record in records:
        action = dict(record)
        action["chunk_vector"] = action.pop("embedding")
        yield {
            "_index": index_name,
            "_id": action["chunk_id"],
            "_source": action,
        }


def bulk_index(
    client: Elasticsearch,
    input_path: Path = INPUT_PATH,
    index_name: str = INDEX_NAME,
    batch_size: int = 200,
) -> int:
    start = time.time()
    count = 0
    for ok, _ in helpers.streaming_bulk(
        client,
        _bulk_actions(index_name, iter_vector_records(input_path)),
        chunk_size=batch_size,
        max_retries=2,
    ):
        if ok:
            count += 1

    elapsed = time.time() - start
    print(f"Total indexed: {count}")
    print(f"Time taken: {elapsed:.2f} seconds")
    return count


def main() -> None:
    client = get_client()
    create_index(client)
    if INPUT_PATH.exists():
        bulk_index(client)
    else:
        print(f"Index created. Skipping bulk insert because {INPUT_PATH} does not exist.")


if __name__ == "__main__":
    main()
