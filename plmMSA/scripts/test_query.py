import json
import time

import requests
import torch

url = "http://localhost:31818/search"


def query_vector(file_path, collection_name):
    with open(file_path, "rb") as f:
        test_vector = [torch.load(f).tolist()]

    payload = {
        "vectors": test_vector,
        "k": 3,
        "nprobe": 1,
        "collection_name": collection_name,
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(f"Time for query: {time.time() - start_time:.2f}")
    if response.status_code == 200:
        results = response.json()
        for i, result in enumerate(results):
            print(f"Query Vector {i + 1}:")
            print("num results:", len(result))
            for neighbor in result[:10]:
                print(
                    f"\tID: {neighbor['id']}, Distance: {neighbor['distance']}, {neighbor['sequence']}"
                )
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


# Query for the first vector
query_vector(
    "./example_embedding_path/ankh_contrastive/A0A841HZG6.pt", "ankh_contrastive"
)

# Query for the second vector
query_vector("./example_embedding_path/esm1b/A0A841HZG6.pt", "esm1b")
