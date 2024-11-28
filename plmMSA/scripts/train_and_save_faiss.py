import argparse
import hashlib
import os
import pickle
import random
import time

import faiss
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm


def print_time_taken(start_time, task_name):
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for {task_name}: {time_taken:.2f} seconds")


def load_embeddings_from_fasta_and_path(fasta_path: str, embeddings_path: str):
    data = []
    id_mapping = {}
    with open(fasta_path, "r") as file:
        sequence = ""
        id = None
        for line in tqdm(file, desc="Loading datasets"):
            if line.startswith(">"):
                if id is not None and sequence:
                    embedding_file = os.path.join(embeddings_path, f"{id}.pt")
                    if os.path.exists(embedding_file):
                        embedding = torch.load(embedding_file).numpy()
                        data.append(embedding)
                        id_mapping[encode_string_to_int(id)] = {
                            "id": id,
                            "sequence": sequence,
                        }
                id = line[1:].strip()
                sequence = ""
            else:
                sequence += line.strip()
        if id is not None and sequence:
            embedding_file = os.path.join(embeddings_path, f"{id}.pt")
            if os.path.exists(embedding_file):
                embedding = torch.load(embedding_file).numpy()
                data.append(embedding)
                id_mapping[encode_string_to_int(id)] = {
                    "id": id,
                    "sequence": sequence,
                }
    return data, id_mapping


def encode_string_to_int(s: str) -> int:
    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()
    unique_int = int(hex_dig, 16) & 0x7FFFFFFFFFFFFFFF  # 양수로 만듦
    return unique_int


def main(input_embeddings_path, output_path, input_fasta, nlist):
    start_time = time.time()

    data, id_mapping = load_embeddings_from_fasta_and_path(
        input_fasta, input_embeddings_path
    )

    # sample_size = int(0.1 * len(data))
    # sampled_data = random.sample(data, sample_size)

    # num_vectors = len(sampled_data)
    # print(f"num_vectors: {num_vectors}")
    # vector_dim = len(sampled_data[0])
    # print(f"vector_dim: {vector_dim}")

    num_vectors = len(data)
    print(f"num_vectors: {num_vectors}")
    vector_dim = len(data[0])
    print(f"vector_dim: {vector_dim}")

    training_sequences = np.array(data, dtype="float32")

    faiss.normalize_L2(training_sequences)

    print_time_taken(start_time, "data preparation and normalization")

    start_time = time.time()

    d = vector_dim
    print(f"nlist: {nlist}")

    quantizer = faiss.IndexFlatIP(d)

    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )

    print_time_taken(start_time, "index setup")

    start_time = time.time()

    index.train(training_sequences)

    print_time_taken(start_time, "index training")

    all_data = np.array(data, dtype="float32")
    faiss.normalize_L2(all_data)
    ids = np.array([id_hex for id_hex in id_mapping.keys()], dtype=np.int64)
    index.add_with_ids(all_data, ids)

    output_db_path = output_path + ".faiss"
    output_id_mapping_path = output_path + "_id_mapping.pkl"
    faiss.write_index(index, output_db_path)

    # Mapping 저장
    with open(output_id_mapping_path, "wb") as f:
        pickle.dump(id_mapping, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save FAISS index.")
    parser.add_argument(
        "--input_embeddings_path",
        "-i",
        type=str,
        required=True,
        help="Input folder containing embeddings.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
        help="Output folder to save the FAISS index.",
    )
    parser.add_argument(
        "--input_fasta", "-f", type=str, required=True, help="Input fasta file."
    )
    parser.add_argument(
        "--nlist",
        "-n",
        type=int,
        required=True,
        help="Number of clusters for the FAISS index.",
    )

    args = parser.parse_args()
    main(args.input_embeddings_path, args.output_path, args.input_fasta, args.nlist)
