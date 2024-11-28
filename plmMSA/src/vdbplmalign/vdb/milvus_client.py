from typing import Any, Dict, List

from pymilvus import Collection, connections

from vdbplmalign.vdb.vdb_abstract import QueryParams, VDBClient


class MilvusClient(VDBClient):
    def __init__(self, host, port):
        if host == "disable" or port == "disable":
            print("Milvus Connection Disabled")
        else:
            connections.connect("default", host=host, port=port)

    def search(self, collection_name: str, query, query_params: QueryParams):
        collection = Collection(name=collection_name)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": query_params.nprobe},
        }
        return collection.search(
            anns_field="sequence_embedding",
            data=[query],
            limit=query_params.limit,
            param=search_params,
            output_fields=["id", "sequence"],
        )

    @staticmethod
    def prepare_results(
        search_result: List[Dict[str, Any]], output_fields: List[str]
    ) -> Dict[str, Any]:
        results = {"query_embedding": {}, "results": {}}
        for hit in search_result:
            result_data = {
                field: hit["metadata"].get(field)
                for field in output_fields
                if field in hit["metadata"]
            }
            result_data["score"] = hit["distance"]
            results["results"][hit["id"]] = result_data
        return results
