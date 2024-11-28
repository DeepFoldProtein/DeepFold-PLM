import hashlib
import logging
import pickle
import traceback
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# FastAPI application
app = FastAPI()


# Function to encode a string to a unique integer
def encode_string_to_int(s: str) -> int:
    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()
    unique_int = int(hex_dig, 16) & 0x7FFFFFFFFFFFFFFF  # Make it positive
    return unique_int


# Function to load true_id mapping from a pickle file
def load_true_id_mapping(mapping_file):
    with open(mapping_file, "rb") as f:
        true_id_mapping = pickle.load(f)
    return true_id_mapping


# Function to query the FAISS index
def query_embeddings(index, query_vector, k=5):
    D, I = index.search(query_vector, k)
    return D, I


# Paths for FAISS index and true_id mappings
faiss_index_paths = {
    "esm1b": "example_vdb_path/esm1b.faiss",
    "ankh_contrastive": "example_vdb_path/ankh_contrastive.faiss",
}
true_id_mapping_paths = {
    "esm1b": "example_vdb_path/esm1b_id_mapping.pkl",
    "ankh_contrastive": "example_vdb_path/ankh_contrastive_id_mapping.pkl",
}

# Load FAISS indices and true_id mappings
indices = {}
true_id_mappings = {}
for collection_name, index_path in faiss_index_paths.items():
    indices[collection_name] = faiss.read_index(index_path)
    indices[collection_name].nprobe = 100
    true_id_mappings[collection_name] = load_true_id_mapping(
        true_id_mapping_paths[collection_name]
    )
    print(f"Loaded index and mapping for {collection_name}")


class SearchRequest(BaseModel):
    vectors: List[List[float]]
    k: Optional[int] = 3000
    nprobe: Optional[int] = 100
    collection_name: str


class Neighbor(BaseModel):
    id: str
    distance: float
    sequence: Optional[str] = None


@app.post("/search", response_model=List[List[Neighbor]])
def search(request: SearchRequest):
    vectors = request.vectors
    k = request.k
    nprobe = request.nprobe
    collection_name = request.collection_name

    if not vectors or not isinstance(vectors, list):
        raise HTTPException(status_code=400, detail="Invalid input format")
    elif collection_name not in indices:
        raise HTTPException(status_code=400, detail="Invalid collection name")
    else:
        collections_to_search = [collection_name]

    try:
        results = []
        for collection in collections_to_search:
            index = indices[collection]
            index.nprobe = nprobe

            query_vectors = np.array(vectors).astype("float32")
            faiss.normalize_L2(query_vectors)
            distances, neighbor_indices = query_embeddings(index, query_vectors, k)

            true_id_mapping = true_id_mappings[collection]
            for i in range(len(vectors)):
                neighbor_metadatas = [
                    true_id_mapping[idx] for idx in neighbor_indices[i]
                ]
                result = [
                    {
                        "id": neighbor_meta["id"],
                        "distance": dist,
                        "sequence": neighbor_meta["sequence"],
                    }
                    for dist, neighbor_meta in zip(distances[i], neighbor_metadatas)
                ]
                results.append(result)

        return results

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=31818)
