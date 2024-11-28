import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from vdbplmalign.config import GlobalConfig
from vdbplmalign.util import log_time
from vdbplmalign.vdb.faiss_client import FaissClient
from vdbplmalign.vdb.models import AnkhModel, BaseModel, ESM1bModel
from vdbplmalign.vdb.vdb_abstract import QueryParams

config = GlobalConfig()


class FetchVdb:
    def __init__(self, ankh_device, esm1b_device):
        self.ankh_device = ankh_device
        self.esm1b_device = esm1b_device
        self.faiss_client = FaissClient(config.FAISS_API_URL)

        self.ankh_model = None
        self.esm1b_model = None

        self.load_models()
        self.warmup_models()

    def load_models(self):
        if not self.ankh_device == "disable":
            self.ankh_model = AnkhModel(
                "/gpfs/deepfold/users/baehanjin/work/Protein-Contrastive/params/ankh_large_full_contrastive_cards_0__3_temp_005_cont2/checkpoint-2500",
                self.ankh_device,
            )
        else:
            Warning("Ankh is disabled.")
        if not self.esm1b_device == "disable":
            self.esm1b_model = ESM1bModel(self.esm1b_device)
        else:
            Warning("Esm1b is disabled.")

    def warmup_models(self):
        # To do
        warmup_sequence = "MKTIIALSYIFCLVFADYKDDDDK"
        self.ankh_model.encode(warmup_sequence, device=self.ankh_device)
        self.esm1b_model.encode(warmup_sequence, device=self.esm1b_device)
        pass

    def search(self, model_name: str, client: FaissClient, query_params: QueryParams):
        input_sequence = query_params.input_sequence
        model: BaseModel = getattr(self, f"{model_name}_model")
        device = getattr(self, f"{model_name}_device")

        # Only apply chunking for ESM1b model and sequences longer than 1022
        if model_name == "esm1b" and len(input_sequence) > 1022:
            chunks = self.chunk_sequence(input_sequence)
            results = {"query_embedding": {}, "results": {}}

            # Calculate limit per chunk
            per_chunk_limit = max(1, query_params.limit // len(chunks))

            for chunk in chunks:
                # Create new query params with adjusted limit for this chunk
                chunk_params = QueryParams(
                    input_sequence=chunk,
                    collection_name=query_params.collection_name,
                    nprobe=query_params.nprobe,
                    limit=per_chunk_limit,
                )

                # Get embedding and search for this chunk
                query = model.encode(chunk, device).squeeze().numpy()
                chunk_results = client.search(
                    chunk_params.collection_name, query, chunk_params
                )
                chunk_results = client.prepare_results(chunk_results)

                # Merge results
                results["query_embedding"].update(chunk_results["query_embedding"])
                for key, value in chunk_results["results"].items():
                    if key in results["results"]:
                        # Keep the better score if sequence already exists
                        if value["score"] > results["results"][key]["score"]:
                            results["results"][key] = value
                    else:
                        results["results"][key] = value

            return results
        else:
            # Original behavior for non-ESM1b models or shorter sequences
            query = model.encode(input_sequence, device).squeeze().numpy()
            res = client.search(query_params.collection_name, query, query_params)
            return client.prepare_results(res)

    def destroy_models(self):
        import gc

        # if self.ankh_model:
        #     self.ankh_model.cpu()
        #     del self.ankh_model
        #     self.ankh_model = None
        # if self.esm1b_model:
        #     self.esm1b_model.cpu()
        #     del self.esm1b_model
        #     self.esm1b_model = None
        # gc.collect()
        # torch.cuda.empty_cache()

        print("Successfully destroyed the models and cleared GPU cache.")

    def search_models(self, input_sequence, nprobe, limit):
        client = self.faiss_client
        query_params_map = {
            "ankh": QueryParams(
                input_sequence=input_sequence,
                collection_name="ankh_contrastive",
                nprobe=nprobe,
                limit=limit,
            ),
            "esm1b": QueryParams(
                input_sequence=input_sequence,
                collection_name="esm1b",
                nprobe=nprobe,
                limit=limit,
            ),
        }

        # results = {}
        # for model_name, query_params in query_params_map.items():
        #     try:
        #         result = self.search(model_name, client, query_params)
        #         results[model_name] = result
        #     except Exception as exc:
        #         logging.error(f"{model_name} generated an exception: {exc}")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.search,
                    model_name,
                    client,
                    query_params,
                ): model_name
                for model_name, query_params in query_params_map.items()
            }

            results = {}
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as exc:
                    logging.error(f"{model_name} generated an exception: {exc}")

        if config.DESTROY_MODELS_AFTER_SEARCH:
            self.destroy_models()

        return results

    @staticmethod
    def chunk_sequence(sequence, chunk_size=1022, slide_size=512):
        forward_chunks, backward_chunks = [], []
        start_index = 0

        # Forward chunks
        while start_index < len(sequence):
            chunk = sequence[start_index : start_index + chunk_size]
            forward_chunks.append(chunk)
            start_index += slide_size

        # Backward chunks
        end_index = len(sequence)
        while end_index > 0:
            chunk = sequence[max(0, end_index - chunk_size) : end_index]
            backward_chunks.append(chunk)
            end_index -= slide_size

        backward_chunks.reverse()
        return forward_chunks + backward_chunks

    @staticmethod
    def append_results(collection_result, result_dict):
        for hits in collection_result:
            for hit in hits:
                if hit.id not in result_dict["results"]:
                    result_dict["results"][hit.id] = {"score": hit.score}

    @staticmethod
    def update_results_with_sequences(result_dict, sequences_from_ankh):
        to_delete = []
        for uniprotAccession, details in result_dict["results"].items():
            if uniprotAccession in sequences_from_ankh:
                details["sequence"] = sequences_from_ankh[uniprotAccession]
            else:
                to_delete.append(uniprotAccession)

        for uniprotAccession in to_delete:
            del result_dict["results"][uniprotAccession]
